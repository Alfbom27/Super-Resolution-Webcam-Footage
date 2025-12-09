import torch
import torch.nn as nn
from torchvision import models, transforms

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    def __init__(self, shift=0):
        super().__init__()

        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.shift = shift
        self.training = False
        self.model.eval()
        self.model.requires_grad_ = False

    def train(self, mode=True):
        self.training = mode

    def forward(self, pred, target):
        sep = pred.shape[0]
        batch = torch.cat([pred, target])
        if self.shift and self.training:
            padded = nn.functional.pad(batch, [self.shift] * 4)
            batch = transforms.RandomCrop(batch.shape[2:])(padded)
        features = self.model(self.normalize(batch))
        x_hat_features, x_features = features[sep:], features[:sep]
        return nn.functional.mse_loss(x_hat_features, x_features)


class CombinedLoss(nn.Module):
    def __init__(self, pixel_loss=None, perceptual_loss=None, pixel_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.perceptual_loss = perceptual_loss
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, pred, target):
        return self.pixel_weight*self.pixel_loss(pred, target) + self.perceptual_weight*self.perceptual_loss(pred, target)

