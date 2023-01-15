import torch
import numpy as np
import os
import json
import string
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def tensor_to_name(tensor):
    name = ""
    for char in tensor:
        try: 
            char_index = (char == 1).nonzero(as_tuple=True)[0].item()
        except ValueError:
            return name
        name = name + all_letters[char_index]

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
                        data[:, 0] == "English"],
                axis=0)]

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
    x_data = [name_to_tensor(i) for i in x_data]

    # Factorise the target
    countries_to_factor_map = {}
    countries_to_factor_map["Arabic"] = torch.Tensor([1, 0, 0])
    countries_to_factor_map["Russian"] = torch.Tensor([0, 1, 0])
    countries_to_factor_map["English"] = torch.Tensor([0, 0, 1])

    y_data = [countries_to_factor_map[i] for i in y_data]

    return x_data, y_data


def load_data_heise():
    torch.set_printoptions(sci_mode=False)
    with open("data/names_heise/transformed_names_heise.json") as file:
        data = json.load(file)
    x_data = np.array(list(data.keys()))
    y_data = np.array(list(data.values()), dtype=object)

    # The data can have multiple occurence scores for a name.
    # Therefore, we would like to keep the country that is the most important
    # for a name.
    for idx, array in enumerate(y_data):
        # For every country we are going to take note of the occurence and
        # index
        occurences = [list(i.values())[0] for i in array]
        max_occurence = max(occurences)
        max_index = occurences.index(max_occurence)
        # Now we can use the max_index to select the country from array
        y_data[idx] = list(array[max_index].keys())[0]

    # Shuffle the arrays together
    randomize = np.arange(len(x_data))
    np.random.shuffle(randomize)
    x_data = x_data[randomize]
    y_data = y_data[randomize]

    # One hot encode the name
    x_data = [name_to_tensor(i) for i in x_data]

    # Factorise the target
    factor_map = {}
    unique_countries = np.unique(y_data)
    for idx, country in enumerate(unique_countries):
        tensor = torch.zeros((1, len(unique_countries)), dtype=torch.float)
        tensor[:, idx] = 1
        factor_map[country] = tensor

    y_data = torch.vstack([factor_map[i] for i in y_data])

    # Write factor_map to disk
    country_to_int = {}
    for i, j in factor_map.items():
        country_to_int[i] =  (j[0] == 1).nonzero(as_tuple=True)[0].item()

    with open("data/country_to_int.json", "w") as file:
        json.dump(country_to_int, file)
    with open("data/int_to_country.json", "w") as file:
        json.dump({j:i for i,j in country_to_int.items()}, file)

    return x_data, y_data


class NameDataset(Dataset):
    def __init__(self, x_data, y_data, batch_size):
        self.x_train = x_data[:int(len(x_data)*0.9)]
        self.x_val = x_data[int(len(x_data)*0.9):]

        self.y_train = y_data[:int(len(y_data)*0.9)]
        self.y_val = y_data[int(len(y_data)*0.9):]

        self.len = len(self.x_train)
        self.batch_size = batch_size
        self.max_batch = self.len // self.batch_size


    def __len__(self):
        return len(self.x_data)


    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])


    def get_train_batch(self, batch_num):
        x_batch = self.x_train[batch_num*self.batch_size:
                               (batch_num+1)*self.batch_size]
        y_batch = self.y_train[batch_num*self.batch_size:
                               (batch_num+1)*self.batch_size]

        x_batch = pad_sequence(x_batch).cuda().permute(1, 0, 2)
        y_batch = pad_sequence(y_batch).cuda().t()

        return x_batch, y_batch


    def get_val_batch(self):
        # Get the number of a batch
        max_batch = len(self.x_val) // self.batch_size
        batch_num = torch.randint(low=0, high=max_batch, size=(1, ))

        # Take the batch
        x_batch = self.x_val[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
        y_batch = self.y_val[batch_num*self.batch_size:(batch_num+1)*self.batch_size]

        x_batch = pad_sequence(x_batch).cuda().permute(1, 0, 2)
        y_batch = pad_sequence(y_batch).cuda().t()

        return x_batch, y_batch
