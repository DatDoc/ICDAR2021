import timm

import torch 
from torch import nn

class Classifier(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if "efficientnet" in model_name:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_classes)
        elif "resnext" in model_name:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_classes)
        elif "vit" in model_name:
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_classes)
        elif "nfnet" in model_name:
            n_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Linear(n_features, n_classes)
        else:
            raise Exception("Add your models in Classifier")

    def forward(self, x):
        x = self.model(x)
        return x

