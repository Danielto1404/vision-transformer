import torch.utils.data
import torchvision.datasets
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from model import VITClassifier


def train(model: nn.Module, criterion, optimizer, trainloader, testloader, epochs):
    for e in range(epochs):

        model.train()
        train_loss = 0
        for x, y in tqdm(trainloader):
            optimizer.zero_grad()

            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Train loss: {train_loss / len(trainloader)}")

        model.eval()
        eval_loss = 0
        for x, y in tqdm(testloader):
            with torch.no_grad():
                y_hat = model(x)
                loss = criterion(y_hat, y)

                eval_loss += loss.item()

        print(f"Eval loss: {eval_loss / len(testloader)}")


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    BATCH_SIZE = 32

    trainset = torchvision.datasets.FashionMNIST(
        root='./dataset/mnist',
        train=True,
        download=False,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    testset = torchvision.datasets.FashionMNIST(
        root='./datasets/mnist',
        train=False,
        download=False,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    classes = (
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    )

    model = VITClassifier(
        image_size=(28, 28),
        num_classes=len(classes),
        patch_size=4,
        in_channels=1,
        layers=4,
        heads=4,
        feedforward_dim=64,
        dropout=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, criterion, optimizer, trainloader, testloader, epochs=5)

    torch.save(model, 'model.pth')
    #
    # m = torch.load('model.pth')

    for x, y in testloader:
        with torch.no_grad():
            y_hat = model(x)

            print(y, y_hat.argmax(1))
            break
