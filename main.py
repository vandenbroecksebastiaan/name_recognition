from load_data import load_data
from model import RNN
from train import train

import torch
import torch.nn as nn
# import numpy
# from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    device = torch.cuda.current_device()
    print(torch.cuda.get_device_name(device))
    # Load the data
    x_train, y_train, x_val, y_val = load_data()
    n_categories = 3
    n_letters = 57

    rnn = RNN(input_dim=n_letters, hidden_dim=2**12, layer_dim=1,
              output_dim=n_categories)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    train_loss = train(rnn, optimizer, criterion, x_train, y_train,
                       x_val, y_val)

    plt.scatter(train_loss)


if __name__ == "__main__":
    main()
