import torch
# from torch.nn.utils import clip_grad_norm


def train(rnn, dataset):

    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    max_batch = dataset.max_batch

    for epoch in range(100):
        for idx in range(max_batch):
            # Take an x and y batch
            x_batch, y_batch = dataset.get_batch(idx)

            rnn.train()
            optimizer.zero_grad()

            # Make a prediction and calculate the loss
            output = rnn(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            if idx % 500 == 0:
                max_output = torch.argmax(output).item()
                max_target = torch.argmax(y_batch).item()

                if max_output == max_target:
                    correct = True
                else:
                    correct = False

                print(epoch, idx, "\t", round(loss.item(), 4), "\t", correct)
                # print(output)
                # print(y_batch)
