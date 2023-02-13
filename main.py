import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import FontClassifierModel, FontClassifierModel2, Resnet32
from synth_text_dataset import SynthTextCharactersDatasetRAM
from transforms import char_transform, img_transform, labels_transform


def train_loop(dataloader, model, loss_fn, optimizer, device) -> tuple[int]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            train_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= num_batches
    correct /= size
    print(
        f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}"
    )

    return train_loss, correct


def test_loop(dataloader, model, loss_fn, device) -> tuple[int]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    return test_loss, correct


def main(params):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    print("Loading data...")

    # filename = "Project/SynthTextData_1000.hdf5"
    # filename = "Project/SynthText_train.h5"
    filename = params["dataset_path"]
    # num_of_images = 1000
    num_of_images = params["num_images"]
    train_percentage = params["train_percentage"]
    init_shape = (100, 100)
    permutation = np.random.permutation(num_of_images)
    train_dataset = SynthTextCharactersDatasetRAM(
        filename,
        full_image_transform=img_transform,
        on_get_item_transform=char_transform,
        target_transform=labels_transform,
        end_idx=int(train_percentage * num_of_images),
        shape=init_shape,
        permutation=permutation,
    )
    test_dataset = SynthTextCharactersDatasetRAM(
        filename,
        full_image_transform=img_transform,
        on_get_item_transform=char_transform,
        target_transform=labels_transform,
        start_idx=int(train_percentage * num_of_images),
        shape=init_shape,
        permutation=permutation,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, shuffle=True)

    print("Data Loaded successfully!")

    classifier = params["model"](init_shape, 1).to(device)
    # classifier = Resnet32(init_shape, 1, num_classes=5).to(device)
    lr = params["lr"]
    epochs = params["epochs"]

    loss_fn = params["loss"]()
    optimizer = params["optimizer"](classifier.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    min_avg_loss = np.inf
    max_acc = -np.inf
    test_accuracies = np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    test_avg_losses = np.zeros(epochs)
    train_avg_losses = np.zeros(epochs)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_avg_losses[t], train_accuracies[t] = train_loop(
            train_dataloader, classifier, loss_fn, optimizer, device
        )
        test_avg_losses[t], test_accuracies[t] = test_loop(
            test_dataloader, classifier, loss_fn, device
        )

        if test_avg_losses[t] < min_avg_loss:
            min_avg_loss = test_avg_losses[t]
            max_acc = test_accuracies[t]
            torch.save(classifier.state_dict(), f"models/{params['name']}.pth")
        # if test_accuracies[t] > max_acc:
        #     max_acc = test_accuracies[t]
        #     min_avg_loss = test_avg_losses[t]
        #     torch.save(classifier.state_dict(), f"models/{params['name']}.pth")

    print("Done!")

    plt.figure()

    plt.subplot(121)
    plt.title("Accuracy over time")
    plt.plot(range(epochs), test_accuracies, color="orange", label="Test")
    plt.plot(range(epochs), train_accuracies, color="green", label="Train")
    plt.legend()

    plt.subplot(122)
    plt.title("Average loss over time")
    plt.plot(range(epochs), test_avg_losses, color="orange", label="Test")
    plt.plot(range(epochs), train_avg_losses, color="green", label="Train")
    plt.legend()

    plt.savefig(f"outputs/{params['name']}.png")

    with open(params["results_file"], "a") as res_file:
        res_file.write(
            ",".join([params["name"], str(max_acc), str(min_avg_loss)]) + "\n"
        )

    # torch.save(classifier.state_dict(), "models/model_weights.pth")


if __name__ == "__main__":
    lr = 1e-2
    epochs = 2
    loss = nn.CrossEntropyLoss
    optimizer = torch.optim.SGD
    batch_size = 32
    train_percentage = 0.8
    results_file = "outputs/res.csv"

    with open(results_file, "w") as res_file:
        res_file.write("name,accuracy,loss\n")

    main(
        {
            "lr": lr,
            "epochs": epochs,
            "loss": loss,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "train_percentage": train_percentage,
            "dataset_path": "Project/SynthText_train.h5",
            "num_images": 30520,  # 998,  # 30520
            "results_file": results_file,
            "name": "_".join(
                [
                    "SynthText_train",
                    str(lr),
                    str(epochs),
                    str(loss.__name__),
                    str(optimizer.__name__),
                    str(batch_size),
                    str(train_percentage),
                ]
            ),
        }
    )
