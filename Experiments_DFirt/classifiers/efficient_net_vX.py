import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetvX(nn.Module):
    def __init__(self, version, input_dim=(3, 256, 256)):
        super().__init__()
        en = EfficientNet.from_pretrained('efficientnet-' + version)
        if version == 'b7':
            in_features=2560
        if version == 'b4':
            in_features=1792
        if version == 'b0':
            in_features=1280
        
        en._fc = nn.Linear(in_features=in_features, out_features=5, bias=True)
        self.model = en
        self.version = version
    
    def forward(self, x):
        return self.model(x)
    
    def save(self):
        path = 'models/EfficientNet' + self.version + '.model'
        print('Saving model... %s' % path)
        torch.save(self.model, path)