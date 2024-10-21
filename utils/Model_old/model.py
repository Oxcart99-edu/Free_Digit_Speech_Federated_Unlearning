from torch import nn
import torch.nn.functional as F

class FLNet(nn.Module):
    def __init__(self):
        super(FLNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Adjusted for 64x64 images, after two 2x2 max-poolings
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # Output size: 32x32
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # Output size: 16x16
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


