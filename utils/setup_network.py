from models.hub import (fgw,resnet,vit)
from utils.checkpoint import restore
from utils.logger import Logger
from utils.helper import init_weights, count_parameters

nets = {
    'fgw': fgw.Model,
    'vit': vit.Model,
    'resnet': resnet.Model
}

def setup_network(network, 
                  in_channels, 
                  model_weight=None,
                  num_classes=7, 
                  freeze_layers='all'):
    
    print('model: ', network)
    net = nets[network](in_channels=in_channels, num_classes=num_classes, freeze_layers=freeze_layers)
    # Prepare logger
    logger = Logger()
    if model_weight is None:
        net.apply(init_weights)

    print(f'total trainable parameters: {count_parameters(net)}')

    return logger, net