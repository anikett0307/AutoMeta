import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Image transformation for meta-learning tasks
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_dataset(data_path, batch_size=32):
    dataset = datasets.ImageFolder(root=data_path, transform=image_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader

if __name__== "_main_":
    dataset_path = "../data/HAM10000"
    dataset, loader = load_dataset(dataset_path)
    print(f"Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")