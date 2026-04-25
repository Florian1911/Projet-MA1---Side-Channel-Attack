import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_len: int, n_classes: int = 256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=11, padding=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=11, padding=5)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.pool = nn.MaxPool1d(2)

        with torch.no_grad():
            x = torch.zeros(1, 1, input_len)
            x = self._features(x)
            flat = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def _features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x

    def forward(self, x):
        x = x.unsqueeze(1)   # (B,L) -> (B,1,L)
        x = self._features(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
