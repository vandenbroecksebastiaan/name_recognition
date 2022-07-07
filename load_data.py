import torch
import numpy as np
import os
import string
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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
    tensor = tensor.type(torch.float)
    tensor = torch.squeeze(tensor, dim=1)
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
    english_idx = np.random.randint(len(english_samples),
                                    size=english_difference)

    data = np.vstack((data, arabic_samples[arabic_idx, :]))
    data = np.vstack((data, english_samples[english_idx, :]))

    # Create a train and validation set
    np.random.shuffle(data)
    x_data = data[:, 1]
    y_data = data[:, 0]

    # One hot encode the names
    x_data = np.array([name_to_tensor(i) for i in x_data])

    # Factorise the target
    countries_to_factor_map = {}
    countries_to_factor_map["Arabic"] = torch.Tensor([1, 0, 0])
    countries_to_factor_map["Russian"] = torch.Tensor([0, 1, 0])
    countries_to_factor_map["English"] = torch.Tensor([0, 0, 1])

    y_data = [countries_to_factor_map[i] for i in y_data]

    return x_data, y_data


class NameDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = len(x_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])

    def get_batch(self, batch_num):
        # x_batch = self.x_data[batch_num*500:(batch_num+1)*500]
        # y_batch = self.y_data[batch_num*500:(batch_num+1)*500]

        x_batch = self.x_data
        y_batch = self.y_data

        x_batch = pad_sequence(x_batch)
        y_batch = pad_sequence(y_batch)

        return x_batch, y_batch
