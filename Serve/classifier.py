""" This is the model wrapper """

from torch import nn
from torchvision import models
import pytorch_lightning as pl
import json


class CIFAR10CLASSIFIER(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    model wrapper for cifar10 classification
    """

    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super().__init__()
        self.model_conv = models.resnet50(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        with open("Serve/class_mapping.json") as f:
            class_mapping = json.load(f)
            num_classes = len(class_mapping)
        self.model_conv.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.model_conv(x)
        return out
