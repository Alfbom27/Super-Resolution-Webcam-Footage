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
    def __init__(self):
        super().__init__()

        self.layers = ['3', '8', '15', '22']
        vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        vgg_model.eval()
        vgg_model.requires_grad_ = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        vgg_model.to(self.device)
        self.vgg_layers = nn.ModuleList([vgg_model[:int(l)] for l in self.layers])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, pred, target):
        pred = self.normalize(pred)
        target = self.normalize(target)

        loss = 0.0
        for layer in self.vgg_layers:
            pred_features = layer(pred)
            target_features = layer(target)
            loss += nn.functional.mse_loss(pred_features, target_features)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, pixel_loss=None, perceptual_loss=None, pixel_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.perceptual_loss = perceptual_loss
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, pred, target):
        return self.pixel_weight*self.pixel_loss(pred, target) + self.perceptual_weight*self.perceptual_loss(pred, target)

