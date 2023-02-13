import os
from itertools import product

from torch import nn, optim

import main
from model import FontClassifierModel2

if __name__ == "__main__":
    curr_run_outputs_dir = "SGD_model2_by_loss"
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    if not os.path.isdir(os.path.join("outputs", curr_run_outputs_dir)):
        os.mkdir(os.path.join("outputs", curr_run_outputs_dir))
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir(os.path.join("models", curr_run_outputs_dir)):
        os.mkdir(os.path.join("models", curr_run_outputs_dir))

    lr_options = [1e-2]
    epochs_options = [50]
    loss_options = [nn.CrossEntropyLoss]
    optimizer_options = [optim.SGD]
    batch_size_options = [32]
    train_percentage_options = [0.8]
    model_options = [FontClassifierModel2]
    results_file = f"outputs/{curr_run_outputs_dir}/results.csv"

    with open(results_file, "w") as res_file:
        res_file.write("name,accuracy,loss\n")

    for lr, epochs, loss, optimizer, batch_size, train_percentage, model in product(
        lr_options,
        epochs_options,
        loss_options,
        optimizer_options,
        batch_size_options,
        train_percentage_options,
        model_options,
    ):
        main.main(
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
                "model": model,
                "name": os.path.join(
                    curr_run_outputs_dir,
                    "_".join(
                        [
                            str(model.__name__),
                            str(lr),
                            str(epochs),
                            str(loss.__name__),
                            str(optimizer.__name__),
                            str(batch_size),
                            str(train_percentage),
                        ]
                    ),
                ),
            }
        )
