import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5, inplace=False)

        ## Block 0
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)

        ## Block 1
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        ## Block 2
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        ## Block 3
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        ## Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        ## Block 5
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        ## Block 6
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        ## Block 7
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        ## Block 8
        self.fc1 = nn.Linear(512, 4096)

        ## Block 9
        self.fc2 = nn.Linear(4096, 4096)

        ## Block 10
        self.fc3 = nn.Linear(4096, num_classes)


    def forward(self, x):

        ## Block 0
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        ## Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        ## Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        ## Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        ## Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        ## Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        ## Block 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)

        ## Block 7
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.MaxPool2d(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        ## Block 8
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        ## Block 9
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        ## Block 10
        x = self.fc3(x)

        return x

