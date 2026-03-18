from torchvision import models, transforms 
import torch
class SkinLesionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.model.features.parameters(): #Freezing the model pre-trained features
            param.requires_grad = False
        self.model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=8, bias=True) #changing output to 8 classes

    def forward(self, x):
        return self.model(x)