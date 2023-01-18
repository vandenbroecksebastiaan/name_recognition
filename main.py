from model import model
from load_data import NameDataset, collate_fn
from train import train

import torch
import matplotlib.pyplot as plt
import numpy as np
import json

with open("data/int_to_country.json") as file:
    int_to_country = json.load(file)

EPOCHS = 20

def print_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Cross entropy loss")
    plt.savefig("output/losses.png", dpi=300, bbox_inches="tight")


def main():
    torch.cuda.device("cuda")

    # Load the data
    dataset = NameDataset(reduce=True)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, lengths=[11300, 1413, 1413]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=512,
                                             collate_fn=collate_fn)

    # Train the model
    n_categories = 4
    n_letters = 57

    rnn = model(input_dim=n_letters, output_dim=n_categories).cuda()

    train_losses, val_losses = train(rnn, train_loader, val_loader,
                                     dataset.weights, EPOCHS)

    # Save the model
    torch.save(rnn, "output/model.pt")

    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
