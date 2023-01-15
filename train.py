import torch
# from torch.nn.utils import clip_grad_norm


def train(rnn, dataset):

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    max_batch = dataset.max_batch
    train_loss = []

    rnn.train()
    for epoch in range(10):

        for idx in range(max_batch):
            # Take an x and y batch
            x_batch, y_batch = dataset.get_train_batch(idx)

            # Make a prediction and calculate the loss
            optimizer.zero_grad()
            output = rnn(x_batch)
            # NOTE: y_batch should be an indicator
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            """
            print(loss)
            for i in list(rnn.parameters()):
                print(i.shape, "\n")

            """

            if idx % 100 == 0:

                x_val, y_val = dataset.get_val_batch()
                val_output = rnn(x_val)
                val_loss = criterion(val_output, y_val)
                train_loss.append(loss.item())
                print(epoch, idx, round(loss.item(), 4),
                      round(val_loss.item(), 4))

    return train_loss
