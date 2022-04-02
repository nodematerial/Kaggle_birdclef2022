
from asyncio.log import logger
import gc
import os
import math
import random
import warnings
import yaml

import albumentations as A
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from pathlib import Path
from typing import List
from tqdm import tqdm

from customaugment import *
from sam import SAM
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from catalyst.core import Callback, CallbackOrder, IRunner
from catalyst.dl import Runner, SupervisedRunner, ISupervisedRunner
from catalyst.loggers.wandb import WandbLogger
from catalyst.callbacks.scheduler import SchedulerCallback
from sklearn import model_selection
from sklearn import metrics
from timm.models.layers import SelectAdaptivePool2d
from torch.optim.optimizer import Optimizer
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

import wandb
#!wandb login your_wandb_apikey

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_SILENT'] = 'true'

with open('config.yml', 'r') as yml:
    CFG = yaml.load(yml, Loader=yaml.SafeLoader)


if CFG['DEBUG']:
    CFG['epochs'] = 3


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: str,
                 img_size=224,
                 waveform_transforms=None,
                 period=20,
                 validation=False):
        self.df = df
        self.datadir = Path(datadir)
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]
        second_codes = sample["secondary_labels"]

        y, sr = sf.read(self.datadir / wav_name)
        if len(y.shape) == 2:
            y = y[:,0]

        len_y = len(y)
        if self.period != 'full':
            effective_length = sr * self.period
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                if not self.validation:
                    start = np.random.randint(effective_length - len_y)
                else:
                    start = 0
                #print(y.shape)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                if not self.validation:
                    start = np.random.randint(len_y - effective_length)
                else:
                    start = 0
                y = y[start:start + effective_length].astype(np.float32)

        y = y.astype(np.float32)
        y = np.nan_to_num(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = np.nan_to_num(y)

        labels = np.zeros(len(CFG['target_columns']), dtype=float)

        if ebird_code in CFG['target_columns']:
            label = torch.tensor(CFG['target_columns'].index(ebird_code), dtype = torch.long)
        else:
            label = torch.tensor(CFG['target_columns'].index('nocall'), dtype = torch.long)

        return y, label


def get_transforms(phase: str):
    transforms = CFG['transforms']
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None
        
        
class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, 
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0000001 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=get_device())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HappyWhaleModel(nn.Module):
    def __init__(self, embedding_size = 512, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(CFG['base_model_name'], pretrained=pretrained, in_chans=1)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = ArcMarginProduct(embedding_size, 
                                   CFG["num_classes"],
                                   s=CFG["s"], 
                                   m=CFG["m"], 
                                   easy_margin=CFG["ls_eps"], 
                                   ls_eps=CFG["ls_eps"])
        
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=CFG['n_fft'], 
                                                 hop_length=CFG['hop_length'],
                                                 win_length=CFG['n_fft'], 
                                                 window="hann", center=True, 
                                                 pad_mode="reflect",
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=CFG['sample_rate'], 
                                                 n_fft=CFG['n_fft'],
                                                 n_mels=CFG['n_mels'], 
                                                 fmin=CFG['fmin'], 
                                                 fmax=CFG['fmax'], 
                                                 ref=1.0, 
                                                 amin=1e-10, 
                                                 top_db=None,
                                                 freeze_parameters=True)

    def forward(self, input, labels):
        images = self.spectrogram_extractor(input)  
        images = self.logmel_extractor(images)
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding, labels)
        return output
    
    def extract(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding


__CRITERIONS__ = {
}


def get_criterion():
    if hasattr(nn, CFG['loss_name']):
        return nn.__getattribute__(CFG['loss_name'])(**CFG['loss_params'])
    elif __CRITERIONS__.get(CFG['loss_name']) is not None:
        return __CRITERIONS__[CFG['loss_name']](**CFG['loss_params'])
    else:
        raise NotImplementedError


# Custom optimizer
__OPTIMIZERS__ = {
}


def get_optimizer(model: nn.Module):
    optimizer_name = CFG['optimizer_name']
    if optimizer_name == "SAM":
        base_optimizer_name = CFG['base_optimizer']
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **CFG['optimizer_params'])

    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **CFG['optimizer_params'])
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                      **CFG['optimizer_params'])


def get_scheduler(optimizer):
    scheduler_name = CFG['scheduler_name']

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **CFG['scheduler_params'])


def display(logger):
    logger.info("=" * 60)
    logger.info('MODEL CONFIG')
    logger.info('memo : {}'.format(CFG['memo']))
    logger.info('')
    logger.info('- model_name : {}'.format(CFG['base_model_name']))
    logger.info('- epochs : {}'.format(CFG['epochs']))
    logger.info('- optimizer : {}'.format(CFG['optimizer_name']))
    logger.info('- criterion : {}'.format(CFG['loss_name']))
    train_augments = []
    valid_augments = []
    for augment in CFG['transforms']['train']:
        train_augments.append(augment['name'])
    for augment in CFG['transforms']['valid']:
        valid_augments.append(augment['name'])
    logger.info(f'- train_augmentations :{train_augments}')
    logger.info(f'- valid_augmentations :{valid_augments}')

    logger.info("=" * 60)

def f1_score(y, clipwise_output, threshold = 0.2):
    pred = clipwise_output > threshold
    return metrics.f1_score(y, pred, average="samples")


def training(logger, fold):
    exp_num= os.path.basename(os.getcwd())
    device = get_device()
    train = pd.read_csv(CFG['train_csv'])
    #train = train[train['primary_label'].isin(CFG['target_columns'])].reset_index(drop=True)
    train.secondary_labels = train.secondary_labels.apply(eval)
    splitter = getattr(model_selection, CFG['split'])(**CFG['split_params'])
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
        if fold not in CFG['folds']:
            continue
        
        logger.info(f"***** Fold {fold} Training *****")
        
        if CFG['wandb']:
            wandbrun = wandb.init(project = CFG['project_name'], 
                            name = exp_num, reinit=True)

        if CFG['DEBUG']:
            trn_df = train.loc[trn_idx, :][0:100].reset_index(drop=True)
            val_df = train.loc[val_idx, :][0:100].reset_index(drop=True)
        else:
            trn_df = train.loc[trn_idx, :].reset_index(drop=True)
            val_df = train.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: torchdata.DataLoader(
                WaveformDataset(
                    df_,
                    CFG['train_datadir'],
                    img_size=CFG['img_size'],
                    waveform_transforms=get_transforms(phase),
                    period=CFG['period'],
                    validation=(phase == "valid")
                ),
                **CFG['loader_params'][phase])  # type: ignore
            for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
        }

        model = HappyWhaleModel().to(device)
        criterion = get_criterion()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)
        scaler = GradScaler()
        mixupper = Mixup(p=CFG['mix_p'], alpha=CFG['mix_alpha'])
        best_loss = 100000
        
        if CFG['load_weight']:
            weight = torch.load(CFG['logdir'] + f'/fold{fold}/best.pth')
            model.load_state_dict(weight)

        for epoch in range(CFG['epochs']):
            # training
            sum_loss = 0
            sum_acc = 0
            model.train()
            for x, y in tqdm(loaders['train']):
                #mixupper.init_lambda()
                x, y = x.to(device), y.to(device)
                #x = model.spectrogram_extractor(x)
                #x = mixupper.lam * x + (1 - mixupper.lam) * x.flip(0)
                #y = mixupper.lam * y + (1 - mixupper.lam) * y.flip(0)
                if CFG['optimizer_name'] == 'SAM':
                    # first forward-backward pass
                    # use this loss for any training statistics
                    loss = criterion(model(x, y), y)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward pass
                    criterion(model(x, y), y).backward()
                    optimizer.second_step(zero_grad=True)
                    sum_loss += loss.item()
                else: 
                    loss = criterion(model(x, y), y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    sum_loss += loss.item()
            scheduler.step()
            train_loss = sum_loss / len(loaders['train'])

            # validation
            sum_loss = 0
            sum_f1s = [0] * len(CFG['thresholds'])
            thresholds = CFG['thresholds']
            main_thresh_id = CFG['main_thresh_id']
            model.eval()
            for x, y in tqdm(loaders['valid']):
                with torch.no_grad():
                    x, y = x.to(device), y.to(device)
                    #x = model.spectrogram_extractor(x)
                    logits = model(x, y)
                    loss = criterion(logits, y)
                    sum_loss+=loss.item()
                    pred = torch.argmax(logits, dim = 1)
                    sum_acc += accuracy_score(pred.to('cpu'), y.to('cpu'))
            valid_loss = sum_loss / len(loaders['valid'])
            valid_acc = sum_acc / len(loaders['valid'])

            if valid_loss < best_loss:
                best_epoch = epoch
                logdir = CFG['logdir']
                torch.save(model.state_dict(), f'{logdir}/fold{fold}/best.pth')
            logdict = {'train/loss' : train_loss, 'valid/loss' : valid_loss, 'valid/accuracy': valid_acc, 'lr':scheduler.get_last_lr()[0]}

            if CFG['wandb']:
                wandb.log(logdict)
            logger.info(f'train_loss:{train_loss:.4g} | valid_loss:{loss:.4g} | accuracy : {valid_acc}')

        logger.info(f'fold{fold}\'s best_score is {best_loss:.4g} at epoch {best_epoch}')

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        
        if CFG['wandb']:
            wandbrun.finish()


def main():
    warnings.filterwarnings("ignore")
    logdir = Path(CFG['logdir'])
    logdir.mkdir(exist_ok=True, parents=True)
    if (logdir / "train.log").exists():
        os.remove(logdir / "train.log")
    set_seed(CFG['seed'])
    logger = init_logger(log_file=logdir / "train.log")

    # display
    display(logger)

    for fold in CFG['folds']:
        foldpath = logdir / f'fold{fold}'
        foldpath.mkdir(exist_ok=True, parents=True)
        training(logger, fold)


if __name__== '__main__':
    main()