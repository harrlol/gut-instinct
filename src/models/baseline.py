import torch.nn as nn

# first try baseline, using cnn + linear regression
class CNNPlusLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),   
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.linear_head = nn.Linear(128, 460)
    
    def forward(self, x):
        x = self.features(x)
        x = self.encoder(x)
        x = self.linear_head(x)
        return x