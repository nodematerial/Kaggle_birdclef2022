
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
        else:
            y = y.astype(np.float32)

        y = np.nan_to_num(y)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        y = np.nan_to_num(y)

        labels = np.zeros(len(CFG['target_columns']), dtype=float)
        labels[CFG['target_columns'].index(ebird_code)] = 1.0
        if CFG['second_label']:
            for code in second_codes:
                labels[CFG['target_columns'].index(code)] = 1.0

        return  y, labels


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


class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class MulBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CFG['num_classes'], 2048, bias=True)
        self.fc2 = nn.Linear(2048, 2048, bias=True)
        self.fc3 = nn.Linear(2048, 2048, bias=True)

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        middle = self.fc1(weight)
        mu = torch.unsqueeze(torch.unsqueeze(self.fc2(middle), 0), 0)
        sigma = torch.unsqueeze(torch.unsqueeze(self.fc3(middle), 0), 0)
        x = x.transpose(1, 3)
        x = torch.mul(x, sigma)
        mu.expand(list(x.size()))
        res = torch.add(x, mu).transpose(1, 3)
        return res


class PANNsCNN14Att(nn.Module):
    def __init__(self):
        super().__init__()

        self.interpolate_ratio = 32  # Downsampled ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=CFG['n_fft'], hop_length=CFG['hop_length'],
                                                 win_length=CFG['n_fft'], window="hann", center=True, pad_mode="reflect",
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=CFG['sample_rate'], n_fft=CFG['n_fft'],
                                                 n_mels=CFG['n_mels'], fmin=CFG['fmin'], fmax=CFG['fmax'], ref=1.0, amin=1e-10, top_db=None,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(CFG['n_mels'])

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.mul_block = MulBlock()

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = AttBlockV2(2048, CFG['num_classes'], activation='sigmoid')

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
    def cnn_feature_extractor(self, x: torch.Tensor, weight: torch.Tensor):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.mul_block(x, weight)
        return x        

    def forward(self, input: torch.Tensor, weight: torch.Tensor):
        # (batch_size, 1, time_steps, freq_bins)
        #x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(input)    # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # ここまで来てる
        # Output shape (batch size, channels, time, frequency)
        x = self.cnn_feature_extractor(x, weight)
        #print(x.size())
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "logit": logit, # logit of clipwise_output
            "framewise_logit": framewise_logit,# logit of framewise_output
            "clipwise_output": clipwise_output
        }

        return output_dict


class SoundEventTriageLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, weight):
        preds = logits["logit"]
        loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=weight)(preds, targets)
        #loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        loss = loss.mean()
        
        return loss


__CRITERIONS__ = {
    "SoundEventTriageLoss": SoundEventTriageLoss
}


def get_criterion():
    if hasattr(nn, CFG['loss_name']):
        return nn.__getattribute__(CFG['loss_name'])(**CFG['loss_params'])
    elif __CRITERIONS__.get(CFG['loss_name']) is not None:
        return __CRITERIONS__[CFG['loss_name']](**CFG['loss_params'])
    else:
        raise NotImplementedError


# Custom optimizer
__OPTIMIZERS__ = {}


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
    logger.info('model config')
    logger.info('- model_name : {}'.format(CFG['base_model_name']))
    logger.info('- epochs : {}'.format(CFG['epochs']))
    logger.info('- optimizer : {}'.format(CFG['optimizer_name']))
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


def training(logger):
    exp_num= os.path.basename(os.getcwd())
    device = get_device()
    train = pd.read_csv(CFG['train_csv'])
    train.secondary_labels = train.secondary_labels.apply(eval)
    splitter = getattr(model_selection, CFG['split'])(**CFG['split_params'])
    for fold, (trn_idx, val_idx) in enumerate(splitter.split(train, y=train["primary_label"])):
        if fold not in CFG['folds']:
            continue
        
        logger.info(f"***** Fold {fold} Training *****")

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

        model = PANNsCNN14Att().to(device)

        criterion = get_criterion()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)
        scaler = GradScaler()
        mixupper = Mixup(p=CFG['mix_p'], alpha=CFG['mix_alpha'])
        best_f1 = 0

        for epoch in range(CFG['epochs']):
            # training
            sum_loss =0
            model.train()
            weight = np.random.dirichlet([1] * CFG['num_classes'])
            for x, y in tqdm(loaders['train']):
                mixupper.init_lambda()
                x, y = x.to(device), y.to(device)
                x = model.spectrogram_extractor(x)
                x = mixupper.lam * x + (1 - mixupper.lam) * x.flip(0)
                y = mixupper.lam * y + (1 - mixupper.lam) * y.flip(0)
                weight_ = np.where(weight > 0.5, weight * CFG['num_classes'], 1)
                weight_ = torch.tensor(weight_).float().to(device)
                if CFG['optimizer_name'] == 'SAM':
                    # first forward-backward pass
                    # use this loss for any training statistics
                    loss = criterion(model(input = x, weight = weight_), y, weight_)
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    # second forward-backward pass
                    criterion(model(x, weight_), y, weight_).backward()
                    optimizer.second_step(zero_grad=True)
                    sum_loss += loss.item()
                else: 
                    loss = criterion(model(input = x, weight = weight_), y, weight_)
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
                    x = model.spectrogram_extractor(x)
                    weight_ = np.where(weight > 0.5, weight * CFG['num_classes'], 1)
                    weight_ = torch.tensor(weight_).float().to(device)
                    logits = model(x, weight_)
                    loss = criterion(logits, y, weight_)
                    sum_loss+=loss.item()
                    for i, threshold in enumerate(thresholds):
                        sum_f1s[i] += f1_score(y.to('cpu'), logits['clipwise_output'].to('cpu'),
                                               threshold = threshold)

            valid_loss = sum_loss / len(loaders['valid'])
            f1s = [ k / len(loaders['valid']) for k in sum_f1s]
            if f1s[main_thresh_id] >= best_f1:
                best_f1 = f1s[main_thresh_id]
                best_epoch = epoch
                torch.save(model.state_dict(), f'out/fold{fold}/best.pth')
            logdict = {'train/loss' : train_loss, 'valid/loss' : valid_loss, 'lr':scheduler.get_last_lr()[0]}
            f1_dict = {}
            for i, threshold in enumerate(thresholds):
                logdict[f'f1_at_{threshold}'] = f1s[i]
                f1_dict[f'f1_at_{threshold}'] = round(f1s[i], 4)

            logger.info(f'train_loss:{train_loss:.4g} | valid_loss:{loss:.4g} | f1_score : {f1_dict}')
            wandb.log(logdict)

        logger.info(f'fold{fold}\'s best_score is {best_f1:.4g} at epoch {best_epoch}')

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        wandbrun.finish()


def main():
    warnings.filterwarnings("ignore")
    logdir = Path(CFG['logdir'])
    logdir.mkdir(exist_ok=True, parents=True)
    for fold in CFG['folds']:
            foldpath = logdir / f'fold{fold}'
            foldpath.mkdir(exist_ok=True, parents=True)
    if (logdir / "train.log").exists():
        os.remove(logdir / "train.log")
    set_seed(CFG['seed'])
    logger = init_logger(log_file=logdir / "train.log")

    # display
    display(logger)

    # main_loop
    training(logger)


if __name__== '__main__':
    main()