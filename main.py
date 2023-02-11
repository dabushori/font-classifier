import torch
from torch import nn
from torch.utils.data import DataLoader

from synth_text_dataset import SynthTextCharactersDatasetRAM
from transforms import img_transform, labels_transform, char_transform
from model import FontClassifierModel, Resnet32


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        pred = model(X)
        loss = loss_fn(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f'{correct = }')
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    print('Loading data...')
    
    filename = "Project/SynthText_train.h5"
    num_of_images = 998
    train_dataset = SynthTextCharactersDatasetRAM(
        filename,
        full_image_transform=img_transform,
        on_get_item_transform=char_transform,
        target_transform=labels_transform,
        end_idx=int(0.8 * num_of_images),
    )
    test_dataset = SynthTextCharactersDatasetRAM(
        filename,
        full_image_transform=img_transform,
        on_get_item_transform=char_transform,
        target_transform=labels_transform,
        start_idx=int(0.8 * num_of_images),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True)

    print('Data Loaded successfully!')
    
    init_shape = (200, 100)
    # classifier = FontClassifierModel(init_shape, 1).to(device)
    classifier = Resnet32(init_shape, 1, num_classes=5).to(device)
    lr = 1e-1
    epochs = 3

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(classifier.parameters(), lr=lr)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, classifier, loss_fn, optimizer, device)
        test_loop(test_dataloader, classifier, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
