import torch
import json

with open("data/int_to_country.json") as file:
    int_to_country = json.load(file)

# Construct weights for class imbalance
from load_data import load_data_heise
x_data, y_data = load_data_heise()
class_count = torch.sum(y_data, dim=0)
class_weights = 1 / class_count
class_weights = class_weights.cuda()


def train(rnn, dataset):

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    max_batch = dataset.max_batch
    train_loss = []

    rnn.train()
    for epoch in range(10):

        for idx in range(max_batch):
            print(epoch, idx, end="\r")

            # Take an x and y batch
            x_batch, y_batch = dataset.get_train_batch(idx)

            # Make a prediction and calculate the loss
            optimizer.zero_grad()
            output = rnn(x_batch)
            loss = criterion(output, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:

                x_val, y_val = dataset.get_val_batch()
                val_output = rnn(x_val)

                print(torch.unique(torch.argmax(output, dim=1), return_counts=True))

                val_loss = criterion(val_output, y_val)
                train_loss.append(loss.item())
                print(epoch,
                      "\t", idx,
                      "\t", round(loss.item(), 4),
                      "\t", round(val_loss.item(), 4))

    return train_loss