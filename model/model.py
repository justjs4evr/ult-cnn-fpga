"""The 1,440-parameter ULT-CNN architecture from the project brief."""

from torch import nn


class ULTCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 6, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
