import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


# Perceptual loss