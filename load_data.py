import torch
import numpy as np
import os
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def tensor_to_name(tensor):
    name = ""
    for i in tensor:
        bool_tensor = i == 1
        index = bool_tensor.nonzero(as_tuple=True)[1][0]
        letter = all_letters[index]
        name += letter
    return name


def name_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    tensor = tensor.type(torch.long)
    return tensor


def load_data():
    # Read all the files and collect the names
    data = np.empty((0, 2))
    for file_name in os.listdir("data/names"):
        # Open the file containing the names
        with open("data/names/" + file_name) as file:
            names = file.read()

        # Make a list of the names
        x_data = names.split("\n")
        x_data = [i for i in x_data if i != ""]

        y_data = [file_name[:-4]] * len(x_data)

        to_add = np.column_stack((y_data, x_data))
        data = np.vstack((data, to_add))

    # Do some tranformations
    data = data[np.any([data[:, 0] == "Arabic",
                        data[:, 0] == "Russian",
                        data[:, 0] == "English"]
                , axis=0)]

    # Oversample the minority classes
    keys, counts = np.unique(data[:, 0], return_counts=True)

    arabic_difference = counts[2] - counts[0]
    english_difference = counts[2] - counts[1]

    arabic_samples = data[data[:, 0] == "Arabic"]
    english_samples = data[data[:, 0] == "English"]

    arabic_idx = np.random.randint(len(arabic_samples), size=arabic_difference)
    english_idx = np.random.randint(len(english_samples), size=english_difference)

    data = np.vstack((data, arabic_samples[arabic_idx, :]))
    data = np.vstack((data, english_samples[english_idx, :]))

    print(np.unique(data[:, 0], return_counts=True))

    # Create a train and validation set
    np.random.shuffle(data)
    data_train = data[:int(data.shape[0]*0.8)]
    data_val = data[int(data.shape[0]*0.8):]

    x_train = data_train[:, 1]
    y_train = data_train[:, 0]
    x_val = data_val[:, 1]
    y_val = data_val[:, 0]

    # One hot encode the names
    x_train = np.array([name_to_tensor(i) for i in x_train])
    x_val = np.array([name_to_tensor(i) for i in x_val])

    # Factorise the target
    countries_to_factor_map = {}
    countries_to_factor_map["Arabic"] = torch.Tensor([1, 0, 0])
    countries_to_factor_map["Russian"] = torch.Tensor([0, 1, 0])
    countries_to_factor_map["English"] = torch.Tensor([0, 0, 1])

    # y_train = [countries_to_factor_map[i] for i in y_train]
    # y_train = [torch.reshape(i, (1, 3)) for i in y_train]
    # y_val = [countries_to_factor_map[i] for i in y_val]
    # y_val = [torch.reshape(i, (1, 3)) for i in y_val]

    y_train = [countries_to_factor_map[i] for i in y_train]
    # y_train = [torch.reshape(i, (1, 3)) for i in y_train]
    y_val = [countries_to_factor_map[i] for i in y_val]
    # y_val = [torch.reshape(i, (1, 3)) for i in y_val]

    return x_train, y_train, x_val, y_val
