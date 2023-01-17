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


def collate_fn(tensor):
    x_batch = [i[0] for i in tensor]
    y_batch = [i[1] for i in tensor]
    x_batch = pad_sequence(x_batch).cuda().permute(1, 0, 2)
    y_batch = pad_sequence(y_batch).cuda().t()

    return x_batch, y_batch


class NameDataset(Dataset):
    def __init__(self, reduce=False):
        data = self.__load_data(reduce=reduce)
        self.x_data = data[0]
        self.y_data = torch.Tensor(data[1])
        self.weights = self.__make_weights()

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])

    def __load_data(self, reduce):
        torch.set_printoptions(sci_mode=False)
        with open("data/names_heise/transformed_names_heise.json") as file:
            data = json.load(file)
        x_data = np.array(list(data.keys()))
        y_data = np.array(list(data.values()), dtype=object)

        # Only keep one country per observation
        for idx, array in enumerate(y_data):
            occurences = [list(i.values())[0] for i in array]
            max_occurence = max(occurences)
            max_index = occurences.index(max_occurence)
            y_data[idx] = list(array[max_index].keys())[0]

        if reduce:
            x_data = x_data[np.isin(y_data, ["china", "usa", "italy", "thenetherlands"])]
            y_data = y_data[np.isin(y_data, ["china", "usa", "italy", "thenetherlands"])]

        # Order the names from short to long
        sort_indices = np.argsort([len(i) for i in x_data])
        x_data = x_data[sort_indices]
        y_data = y_data[sort_indices]

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

    def __make_weights(self):
        class_count = torch.sum(self.y_data, dim=0)
        class_weights = min(class_count) / class_count
        class_weights = class_weights.cuda()

        return class_weights