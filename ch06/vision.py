import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

x, label = dataset[0]

print("type:", type(x))
print("shape:", x.shape)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for x, label in dataloader:
    print("x shape:", x.shape)
    print("label shape:", label.shape)
    break
