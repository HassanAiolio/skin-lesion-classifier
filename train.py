# Training script - designed to run on Kaggle Notebooks
# Dataset: https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification

import torch
from torchvision import transforms
from model import SkinLesionClassifier
from dataset import SkinLesionDataset
import os

base_path = "/kaggle/input/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification"
csv_path = os.path.join(base_path, "ISIC_2019_Training_GroundTruth.csv")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), antialias=True),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_t = SkinLesionDataset(csv_path, base_path, train_transform)
dataset_v = SkinLesionDataset(csv_path, base_path, val_transform)

generator = torch.Generator().manual_seed(42)
trainDataset_split = torch.utils.data.random_split(dataset_t, [0.8, 0.2], generator)[0]
generator = torch.Generator().manual_seed(42)
valDataset_split = torch.utils.data.random_split(dataset_v, [0.8, 0.2], generator)[1]

train_loader = torch.utils.data.DataLoader(trainDataset_split, batch_size=64, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(valDataset_split, batch_size=64, shuffle=False, num_workers=4)

model = SkinLesionClassifier()

adam = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
weights = dataset_t.get_class_weights().to(device)
loss = torch.nn.CrossEntropyLoss(weight=weights)


nb_epochs = 10
#Training loop
for epoch in range(nb_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    print(f"Epoch {epoch}:")
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.long()
        adam.zero_grad()
        preds = model(images)
        output = loss(preds, labels)
        output.backward()
        train_loss += output.item()
        adam.step()
        _, class_preds = torch.max(preds, dim=1)
        correct += (class_preds == labels).sum().item()
    print(f"Average loss: {train_loss/len(train_loader)}       Accuracy: {correct/len(trainDataset_split)}")

    #Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            output = loss(preds, labels)
            val_loss += output.item()
            _, class_preds = torch.max(preds, dim=1)
            val_correct += (class_preds == labels).sum().item()
        print(f"Average val_loss: {val_loss/len(val_loader)}       Val_Accuracy: {val_correct/len(valDataset_split)}")

torch.save(model.state_dict(), "/kaggle/working/model.pth")
        