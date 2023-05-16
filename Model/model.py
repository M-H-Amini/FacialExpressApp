import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        n_classes = 4

        vgg19 = torchvision.models.efficientnet_b0(pretrained=True)

        for i, param in enumerate(vgg19.parameters()):
            if i < 130:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.cnn_temp = list(vgg19.children())[:-1]
        self.cnn = torch.nn.Sequential(*self.cnn_temp)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=1280, out_features=512),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(0.5),
                                          torch.nn.Linear(in_features=512, out_features=n_classes),
                                          torch.nn.LogSoftmax(dim=1))
    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


