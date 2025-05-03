import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from DNN_shape_classifier import get_shape_model
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_transform():
    return transform

dataset = datasets.ImageFolder("shape_dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_shape_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(30):
    model.train()
    total_loss = 0
    correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    acc = correct / len(dataset)
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2%}")


torch.save(model.state_dict(), "resnet_shape.pt")

def predict_shape(img_patch, model, transform, device):
    model.eval()
    with torch.no_grad():
        img_tensor = transform(img_patch).unsqueeze(0).to(device)
        output = model(img_tensor)
        return output.argmax(dim=1).item()


