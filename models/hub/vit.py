from torchvision.models import vit_l_32, ViT_L_32_Weights
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=7,
                 *args, **kwargs):
        
        super(Model, self).__init__()
        self.model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
        if in_channels != self.model.conv_proj.in_channels:
            self.model.conv_proj = nn.Conv2d(
                in_channels, 1024, 
                kernel_size=(32, 32), 
                stride=(32, 32)
            )
        if kwargs['freeze_layers'] == 'all':
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.heads.head = nn.Linear(
            in_features=1024,
            out_features=num_classes,
            bias=True
        )

    def count_parameters(self): 
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = Model(3)
    print(model(x).shape)
