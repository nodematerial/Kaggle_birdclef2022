from passt import *
import torch 
#model = PaSST()
model = deit_base_distilled_patch16_384(pretrained=True)
import torch.utils.data as torchdata

class TestDataset(torchdata.Dataset):
    def __init__(self):
        pass
        
    def __len__(self):
        return 100
    
    def __getitem__(self, idx: int):
        return torch.rand(1, 128, 99)

#data = TestDataset()
#model.train()
#loader = torchdata.DataLoader(data, batch_size=32, shuffle=False)

#for tensor in loader:
#    y = model(tensor)[0]
#    print(y.size())
