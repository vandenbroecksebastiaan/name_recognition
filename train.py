import torch
# from torch.nn.utils import clip_grad_norm


def train(rnn, dataset):

    model = rnn
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer, base_lr=0.0001, max_lr=0.05, cycle_momentum=True
    # )
    criterion = torch.nn.CrossEntropyLoss()
    max_batch = dataset.max_batch
    train_loss = []

    rnn.train()
    for epoch in range(10):

        for idx in range(max_batch):
            # Take an x and y batch
            x_batch, y_batch = dataset.get_train_batch(idx)

            # Make a prediction and calculate the loss
            output = rnn(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            """
            print(loss)
            for i in list(rnn.parameters()):
                print(i.shape, "\n")

            """

            if idx % 100 == 0:

                # x_val, y_val = dataset.get_val_batch()
                # val_output = rnn(x_val)
                # val_loss = criterion(val_output, y_val)
                train_loss.append(loss.item())
                print(epoch, idx, round(loss.item(), 4),
                      optimizer.param_groups[0]['lr'], "\t")
                # round(val_loss.item(), 4))
                # print(output[0, :].tolist(),
                #       y_batch[0, :].tolist())

            # scheduler.step()

    return train_loss
