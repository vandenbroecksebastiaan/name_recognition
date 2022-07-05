import torch
from tqdm import tqdm


def train(rnn, optimizer, criterion, x_train, y_train, x_val, y_val):
    train_loss = []
    train_prediction = []

    correct_count = 0
    false_count = 0

    for i in range(len(x_train)):
        # Take an x and y batch
        x_sample = x_train[i]
        y_sample = y_train[i]

        rnn.train()
        optimizer.zero_grad()

        # Make a prediction and calculate the loss
        output = rnn(x_sample)
        loss = criterion(output, y_sample)

        max_output = torch.max(output, 0, keepdim=True)[1][0].item()
        max_target = torch.max(y_sample, 0, keepdim=True)[1][0].item()

        if max_output == max_target:
            eval_ = "correct"
            correct_count += 1
        else:
            eval_ = "false"
            false_count += 1

        print(i, loss, max_output, max_target, eval_, "\t",
              round(correct_count / (correct_count + false_count), 2))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Save the loss
        if i % 10 == 0:
            train_loss.append(loss)
            train_prediction.append(output)

        if torch.isnan(loss):
            print("nan loss detected")
            break

    return train_loss
