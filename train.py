import torch
# from torch.nn.utils import clip_grad_norm


def train(rnn, dataset):

    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.0001, max_lr=0.05, cycle_momentum=True
    )
    criterion = torch.nn.MSELoss()

    max_batch = dataset.max_batch

    for epoch in range(100):

        for idx in range(max_batch // 2):
            # Take an x and y batch
            x_batch, y_batch = dataset.get_train_batch(idx)

            rnn.train()
            optimizer.zero_grad()

            # Make a prediction and calculate the loss
            output = rnn(x_batch)
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

                print(epoch, idx, optimizer.param_groups[0]['lr'], "\t",
                      round(loss.item(), 4),
                      round(val_loss.item(), 4))
                print(output[0, :].tolist(),
                      y_batch[0, :].tolist())

            scheduler.step()
