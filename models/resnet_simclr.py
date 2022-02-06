import torch.nn as nn
import torchvision.models as models
import os 
import torch
from exceptions.exceptions import InvalidBackboneError
from models.chexnet import DenseNet121
from torchinfo import summary


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim,arch_weights=""):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim),
                            "chexnet":  DenseNet121(pretrained=False, num_classes=out_dim)}

        self.arch_weights = arch_weights
        self.backbone = self._get_basemodel(base_model)
        print(self.backbone.state_dict().keys())
        if(base_model=="chexnet"):
            dim_mlp = self.backbone.classifier.in_features
            # add mlp projection head
            self.backbone.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.classifier)
        else:
            dim_mlp = self.backbone.fc.in_features
            # add mlp projection head
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

        summary(self.backbone,input_size=(16,3, 224, 224))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            model = self.load_weights(model_name,model)
            # layer = model.conv1
            # new_layer = nn.Conv2d(in_channels=1, 
            #       out_channels=layer.out_channels, 
            #       kernel_size=layer.kernel_size, 
            #       stride=layer.stride, 
            #       padding=layer.padding,
            #       bias=layer.bias)
            # # new_layer.weight[:, :1, :, :] = layer.weight[:, :1, :, :].clone()
            # new_layer.weight = nn.Parameter(new_layer.weight)
            # model.conv1 = new_layer

        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model
    def load_weights(self,model_name,model):
        if(model_name=="chexnet"):
            if os.path.isfile(self.arch_weights):
                print("=> loading arch weights")
                checkpoint = torch.load(self.arch_weights)
                state_dict = checkpoint['state_dict']
                for key in list(state_dict.keys()):
                    state_dict[key[:7].replace('.1.', '1.'). replace('.2.', '2.')] = state_dict.pop(key)
                model.load_state_dict(state_dict)
                print("=>  arch weights loaded")
            else:
                print("=> no arch weights found")
        return model
    def forward(self, x):
        return self.backbone(x)
