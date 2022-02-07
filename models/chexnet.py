import torch.nn as nn
import torchvision
from torchinfo import summary


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self,pretrained=True, num_classes=14):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        summary(self.densenet121,input_size=(16,3, 224, 224))

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.densenet121(x)
        return x
