import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device, vgg_layers):
        super().__init__()
        assert isinstance(vgg_layers, dict), "vgg_layers should be a dictionary"

        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device)
        self.vgg.eval()
        
        # freeze the model
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.feature_layers = list(vgg_layers.keys())
        self.weights = list(vgg_layers.values())

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize input images using ImageNet statistics"""
        return (image - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using VGG feature maps"""
        fake = self.normalize(fake)
        real = self.normalize(real)

        loss = 0.0
        x, y = fake, real
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.feature_layers:
                idx = self.feature_layers.index(i)
                loss += self.weights[idx] * F.l1_loss(x, y)
                
        return loss