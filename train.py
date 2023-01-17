import torch
import json

with open("data/int_to_country.json") as file:
    int_to_country = json.load(file)


def train(model, train_loader, val_loader, EPOCHS):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(EPOCHS):

        for idx, (x_train, y_train) in enumerate(train_loader):
            print(epoch, idx, end="\r")

            # Make a prediction and calculate the loss
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, torch.argmax(y_train, dim=1))
            loss.backward()
            optimizer.step()

            # Do a validation step
            if idx % 10 == 0:
                idx, (x_val, y_val) = next(enumerate(val_loader))
                val_output = model(x_val)
                val_loss = criterion(val_output, torch.argmax(y_val, dim=1))

                train_losses.append(loss.item())
                val_losses.append(val_loss.item())

                print(epoch,
                      "\t", idx,
                      "\t", round(loss.item(), 4),
                      "\t", round(val_loss.item(), 4))

    return train_losses, val_losses