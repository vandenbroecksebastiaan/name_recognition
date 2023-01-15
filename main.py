from model import RNN
from load_data import NameDataset, load_data, load_data_heise, tensor_to_name
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
    x_data, y_data = load_data_heise()
    dataset = NameDataset(x_data, y_data, batch_size=256)

    n_categories = 55
    n_letters = 57

    rnn = RNN(input_dim=n_letters,
              hidden_dim=2056,
              layer_dim=3,
              output_dim=n_categories).cuda()

    train_loss = train(rnn, dataset)

    plt.plot(range(len(train_loss)), train_loss)
    plt.savefig("output/temp.png")


if __name__ == "__main__":
    main()
