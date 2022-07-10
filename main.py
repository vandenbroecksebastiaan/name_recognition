from model import RNN
from load_data import NameDataset, load_data_heise
from train import train

import torch
import matplotlib.pyplot as plt

# TODO: make an implementation that uses 1d convs


def main():
    torch.cuda.device("cuda")

    # Load the data
    x_data, y_data = load_data_heise()
    dataset = NameDataset(x_data, y_data, batch_size=16)

    n_categories = 55
    n_letters = 57

    rnn = RNN(input_dim=n_letters, hidden_dim=1024, layer_dim=3,
              output_dim=n_categories).cuda()

    """
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, rnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    """

    train_loss = train(rnn, dataset)
    plt.plot(range(len(train_loss)), train_loss)
    plt.show()


if __name__ == "__main__":
    main()
