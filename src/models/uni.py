import torch.nn as nn

class FinetunedUni(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(base_model.num_features, num_classes)
    
    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x