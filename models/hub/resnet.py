from torchvision.models import resnet101, ResNet101_Weights
import torch
import torch.nn as nn

# def count_parameters(model): 
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Model(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=7):
        
        super(Model, self).__init__()
        
        self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        if in_channels != self.model.conv1.in_channels:
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, 
                kernel_size=(7, 7), 
                stride=(2, 2), 
                padding=(3, 3), 
                bias=False
            )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes, bias=True)
    
    def forward(self, x):
        out = self.model(x)
        return out
if __name__ == '__main__':
    x = torch.randn(1, 3, 128, 128)
    model = Model(3)
    print(model(x).shape)
