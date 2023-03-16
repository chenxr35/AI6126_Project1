import torchvision.models as models
import torch.nn as nn
import torch
from torchviz import make_dot

MODEL = 'resnet152'
OUTPUT_DIM = 26

if __name__ == '__main__':

    # load the model
    if MODEL == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V1')
    elif MODEL == 'resnet152':
        model = models.resnet152(weights='IMAGENET1K_V1')

    # replace the pretrained model's linear layer with a linear layer with the required dimensions
    IN_FEATURES = model.fc.in_features

    model.fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)

    sampledata = torch.rand(1, 3, 224, 224)

    out = model(sampledata)
    g = make_dot(out)
    g.render(MODEL, view=False)