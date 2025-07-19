import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        for param in self.cnn.parameters():
            param.requires_grad = train_cnn  # Freeze or train CNN

        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Remove final FC layer
        self.linear = nn.Linear(self.cnn[-1][-1].bn2.num_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.cnn(images).squeeze()
        features = self.linear(features)
        features = self.bn(features)
        return features
