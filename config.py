import torch
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
epochs = 10
lr = 3e-4
train_batch_size = 64
test_batch_size = 128
num_workers = 2

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop((512, 512)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
