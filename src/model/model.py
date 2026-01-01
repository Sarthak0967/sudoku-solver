import torch
import torch.nn as nn
from torchvision import models


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # pooling z rozmiarem 2x2
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1
        )  # padding=1, więc rozmiar pozostaje 14x14
        self.fc1 = nn.Linear(
            64 * 7 * 7, 64
        )  # 64 kanały, po poolingach obraz ma wymiary 7x7
        self.fc2 = nn.Linear(64, num_classes)  # 10 klas

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Warstwa konwolucyjna + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Warstwa konwolucyjna + pooling
        x = x.flatten(start_dim=1)  # Wypłaszczanie (64 kanały, 7x7 rozmiar)
        x = torch.relu(self.fc1(x))  # Warstwa w pełni połączona
        x = self.fc2(x)  # Wyjście
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Try to load torchvision pretrained weights in a way that is compatible
        # with multiple torchvision versions and works offline (falls back to
        # randomly initialized model if weights are unavailable).
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception:
            # Older/newer torchvision versions might not expose ResNet18_Weights
            # or the environment may be offline. Fall back to no weights.
            weights = None

        try:
            self.resnet = models.resnet18(weights=weights)
        except TypeError:
            # For very old torchvision versions that expect `pretrained=` keyword
            # instead of `weights=`, try that signature.
            try:
                self.resnet = models.resnet18(pretrained=(weights is not None))
            except Exception:
                # As a last resort, instantiate without pretrained weights.
                self.resnet = models.resnet18()
        for param in list(self.resnet.parameters())[:-5]:
            param.requires_grad = False

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.resnet(x)
