# import torch
# from tqdm import tqdm


def train(rnn, optimizer, criterion, dataset):

    for idx in range(1):
        # Take an x and y batch
        x_batch, y_batch = dataset.get_batch(idx)

        rnn.train()
        optimizer.zero_grad()

        # Make a prediction and calculate the loss
        output = rnn(x_batch)
        y_batch = y_batch.t()
        loss = criterion(output, y_batch)
        print(loss)

        """
        max_output = torch.max(output, 0, keepdim=True)[1][0].item()
        max_target = torch.max(y_sample, 0, keepdim=True)[1][0].item()

        if max_output == max_target:
            eval_ = "correct"
            correct_count += 1
        else:
            eval_ = "false"
            false_count += 1

        print(idx, loss, max_output, max_target, eval_, "\t",
              round(correct_count / (correct_count + false_count), 2))
        """

        loss.backward()
        print(loss)
        optimizer.step()
