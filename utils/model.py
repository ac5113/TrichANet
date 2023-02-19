import torch
from torchvision import models
import torch.nn as nn

from utils.components import ScaledDotProductAttention

encoder = models.vit_l_32(True)
encoder.heads = nn.GELU()

for param in encoder.parameters():
    param.requires_grad = False 

classifier = nn.Sequential(
    nn.Linear(1024, 256, True), 
    nn.ReLU(),
    nn.Linear(256, 2, True),
    nn.Softmax(1)
    )

class ModelSA(nn.Module):
    def __init__(self):
        super(ModelSA, self).__init__()
        self.encoder = encoder
        self.sa = ScaledDotProductAttention(1, 128)
        self.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)

        bs, c = x.shape
        x = torch.reshape(x, (bs, 1, c, 1))

        x = self.sa(x)
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 2)

        x = self.classifier(x)
        
        return x