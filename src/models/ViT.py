import timm

import torch 
from torch import nn

class Classifier(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'tf_efficientnet_b4_ns':
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, n_class)
        elif model_name == 'resnext50_32x4d':
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, n_class)
        elif model_name == 'vit_base_patch16_384':
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, n_class)
        else:
            raise Exception("Add your models in Classifier")

    def forward(self, x):
        x = self.model(x)
        return x

