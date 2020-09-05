import torch
import torchsummary
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
from torchvision import models

class Resnet(nn.Module):
    def __init__(self, out_size, em_size, model_type, pretrained):
        super(Resnet, self).__init__()
        if model_type == 18:
            m = models.resnet18(pretrained)
        else:
            m = models.resnet152(pretrained)
        for param in m.parameters():
            param.requires_grad = False
        fc_in = m.fc.in_features
        m.fc = nn.Linear(fc_in, out_features=em_size)
        self.base = m
        self.fc2 = nn.Linear(em_size, out_size)

    def forward(self, inputs):
        o = self.base(inputs)
        return self.fc2(o)


def build_model(out_size, em_size, model_type=152, pretrained=True):
    return Resnet(out_size, em_size, model_type, pretrained)


if __name__ == '__main__':
    model_ft = build_model(40, 300)
    torchsummary.summary(model_ft, input_size=(3,256,256), device='cpu')

    for m in model_ft.named_modules():
        print(m)


