import gradio as gr
import torch
import os
from torchvision import transforms, models

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SkinLesionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.model.features.parameters(): #Freezing the model pre-trained features
            param.requires_grad = False
        self.model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=8, bias=True) #changing output to 8 classes

    def forward(self, x):
        return self.model(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
model = SkinLesionClassifier()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

model.to(device)

def predict(image):
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).squeeze().tolist()
        classes = [
        'Melanoma (MEL)',
        'Melanocytic Nevus (NV)',
        'Basal Cell Carcinoma (BCC)',
        'Actinic Keratosis (AK)',
        'Benign Keratosis (BKL)',
        'Dermatofibroma (DF)',
        'Vascular Lesion (VASC)',
        'Squamous Cell Carcinoma (SCC)'
    ]
    return {classes[i]: probabilities[i] for i in range(len(classes))}
        
        
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=8),
    title="Skin Lesion Classifier"
).launch()