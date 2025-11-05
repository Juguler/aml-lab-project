"""
CustomNet model for Tiny-ImageNet classification
"""
import torch
from torch import nn


class CustomNet(nn.Module):
    """
    Custom Convolutional Neural Network for Tiny-ImageNet (200 classes)
    Architecture: 5 Conv layers + MaxPool + 1 FC layer
    """
    def __init__(self, num_classes=200):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: B x 64 x 112 x 112

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: B x 128 x 56 x 56

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: B x 256 x 28 x 28

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: B x 512 x 14 x 14

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: B x 512 x 7 x 7

        # Fully connected layer
        self.fc1 = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Forward pass
        Input: B x 3 x 224 x 224
        Output: B x num_classes
        """
        x = self.pool1(self.conv1(x).relu())  # B x 64 x 112 x 112
        x = self.pool2(self.conv2(x).relu())  # B x 128 x 56 x 56
        x = self.pool3(self.conv3(x).relu())  # B x 256 x 28 x 28
        x = self.pool4(self.conv4(x).relu())  # B x 512 x 14 x 14
        x = self.pool5(self.conv5(x).relu())  # B x 512 x 7 x 7

        x = x.view(-1, 512 * 7 * 7)  # Flatten the output for the fully connected layer
        x = self.fc1(x)

        return x
