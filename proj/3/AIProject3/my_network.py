from torch import nn
import torch.nn.functional as F

class MyClassifier(nn.Module):
    def __init__(self) -> None:
        super(MyClassifier, self).__init__()
        self.hidden1 = nn.Linear(256, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x