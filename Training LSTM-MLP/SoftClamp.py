import torch
import torch.nn as nn

class SoftClamp(nn.Module):
    def __init__(self, max_val, soft_val):
        super(SoftClamp, self).__init__()
        self.max_val = max_val
        self.soft_val = soft_val

    def forward(self, x):
        term1 = 0.5 * (-torch.abs(x - self.max_val) + torch.abs(x) + self.max_val)
        term2 = 0.5 * (self.max_val * self.soft_val) * (torch.tanh(2 * x / self.max_val - 1) + 1)
        denominator = self.soft_val + 1
        return (term1 + term2) / denominator