from model import model
from load_data import NameDataset, collate_fn
from train import train

import torch
import matplotlib.pyplot as plt
import numpy as np
import json

with open("data/int_to_country.json") as file:
    int_to_country = json.load(file)

# TODO: make an implementation that uses 1d convs

def print_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def main():
    torch.cuda.device("cuda")

    # Load the data
    dataset = NameDataset()
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, lengths=[35947, 4494, 4494]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1024,
                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024,
                                              collate_fn=collate_fn)

    # Train the model
    n_categories = 55
    n_letters = 57

    rnn = model(input_dim=n_letters,
                hidden_dim=2056,
                layer_dim=3,
                output_dim=n_categories).cuda()

    train_loss = train(rnn, train_loader, val_loader)


if __name__ == "__main__":
    main()
