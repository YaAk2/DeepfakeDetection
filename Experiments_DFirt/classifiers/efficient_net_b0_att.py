import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

    
class LinearAttentionBlock(nn.Module):
        def __init__(self, in_features):
            super(LinearAttentionBlock, self).__init__()
            self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
        def forward(self, l, g):
            N, C, W, H = l.size()
            c = self.op(l+g) #torch.sqrt(l*g + 1e-3) 
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
            g = torch.mul(a.expand_as(l), l)
            g = g.view(N,C,-1).sum(dim=2) 
            return c.view(N,1,W,H), g

    
class EfficientNetB0Att(nn.Module):
    def __init__(self, version, input_dim=(3, 256, 256), reg=False):
        super(EfficientNetB0Att, self).__init__()
        self.reg = reg
        self.version = version
        
        en = EfficientNet.from_pretrained('efficientnet-' + version)
        en._fc = nn.Linear(in_features=1280, out_features=5, bias=True)
    
        self.projector1 = ProjectorBlock(1280, 80)
        self.projector2 = ProjectorBlock(1280, 112)
        self.projector3 = ProjectorBlock(1280, 192)
        self.projector4 = ProjectorBlock(1280, 192)
        self.projector5 = ProjectorBlock(1280, 192)
        self.projector6 = ProjectorBlock(1280, 192)
        self.projector7 = ProjectorBlock(1280, 320)
        self.attn1 = LinearAttentionBlock(in_features=80)
        self.attn2 = LinearAttentionBlock(in_features=112)
        self.attn3 = LinearAttentionBlock(in_features=192)
        self.attn4 = LinearAttentionBlock(in_features=192)
        self.attn5 = LinearAttentionBlock(in_features=192)
        self.attn6 = LinearAttentionBlock(in_features=192)
        self.attn7 = LinearAttentionBlock(in_features=320)
        
        self.model = en
        
    def forward(self, x):
        x = self.model._bn0(self.model._conv_stem(x))
        
        l = []
        for i, block in enumerate(self.model._blocks):
            x = block(x)
            if i==7:
                l.append(x)
            if i==10:
                l.append(x)
            if i==11:
                l.append(x)
            if i==12:
                l.append(x)
            if i==13:
                l.append(x)              
            if i==14:
                l.append(x)
            if i==15:
                l.append(x)
                
        g = self.model._dropout(self.model._avg_pooling(self.model._bn1(self.model._conv_head(x))))
        
        g = F.interpolate(g, scale_factor=l[0].size()[2]/g.size()[2], 
                          mode='bilinear', align_corners=False)
        c1, g1 = self.attn1(l[0], self.projector1(g))
        c2, g2 = self.attn2(l[1], self.projector2(g))
        g = F.interpolate(g, scale_factor=l[2].size()[2]/g.size()[2], 
                  mode='bilinear', align_corners=False)
        c3, g3 = self.attn3(l[2], self.projector3(g))
        c4, g4 = self.attn4(l[3], self.projector4(g))
        c5, g5 = self.attn5(l[4], self.projector5(g))
        c6, g6 = self.attn6(l[5], self.projector6(g))
        c7, g7 = self.attn7(l[6], self.projector7(g))
        g = torch.cat((g1,g2,g3, g4, g5, g6, g7), dim=1)
 
        x = self.model._swish(self.model._fc(g)) 
        
        return x #[x, c1, c2, c3]
    
    def save(self):
        if self.reg==True:
            path = 'models/EfficientNet' + self.version + 'Att_reg' + '.model'
        else:
            path = 'models/EfficientNet' + self.version + 'Att' + '.model'
        print('Saving model... %s' % path)
        torch.save(self, path)