from collections import deque
import string
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import TESNamesDataset
from models import TESLSTM


def generate(race, gender, char, dataset, model, device):
    ''' Generates a "novel" name given the parameters.

    Given the desired race, gender and initial character, the trained model
    will produce a new name by predicting what letter should come next and
    feeding the predicted letter as an input to the model until it reaches
    the maximum length or the terminating character is predicted.

    Parameters
    ----------
    race: str
        Desired race for new name.
    gender: str
        Desired gender for new name.
    char: str
        Starting character of the new name.
    dataset: torch.utils.data.Dataset
        The dataset of Elder Scrolls names.
    model: models.TESLSTM
        The trained model used for prediction.
    device: torch.device
        The device on which to execute.
    '''
    name = char
    model.eval()

    t_race, t_gender, t_char = dataset.one_hot_sample(race, gender, char)
    t_hidden, t_cell = model.init_hidden(1)

    t_race = t_race.view(1, 1, -1).to(device)
    t_gender = t_gender.view(1, 1, -1).to(device)
    t_char = t_char.view(1, 1, -1).to(device)
    t_hidden = t_hidden.to(device)
    t_cell = t_cell.to(device)

    for _ in range(dataset.max_length):
        t_char, t_hidden, t_cell = \
            model(t_race, t_gender, t_char, t_hidden, t_cell)

        char_idx = t_char.argmax(dim=1).item()
        new_char = dataset.char_codec.inverse_transform([char_idx])[0]

        if new_char == '\0':
            break
        else:
            name += new_char
            t_char = dataset.to_one_hot(dataset.char_codec, [new_char])
            t_char = t_char.view(1, 1, -1).to(device)

    return name


if __name__ == '__main__':
    data_root = '/home/syafiq/Data/tes-names/'
    charset = string.ascii_letters + '\'- '
    max_length = 30

    # Prepare GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset.
    dataset = TESNamesDataset(data_root, charset, max_length)

    input_size = (
        len(dataset.race_codec.classes_) +
        len(dataset.gender_codec.classes_) +
        len(dataset.char_codec.classes_)
    )
    hidden_size = 128
    output_size = len(dataset.char_codec.classes_)

    # Prepare model.
    model = TESLSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('model.pt'))
    model = model.to(device)

    new_names = []

    # Predict a name for all combinations.
    for race in dataset.race_codec.classes_:
        for gender in dataset.gender_codec.classes_:
            for letter in string.ascii_uppercase:
                name = generate(race, gender, letter, dataset, model, device)
                print(race, gender, name)
                new_names.append(name)

    # See how many names are copied from the dataset, if any.
    sample_names = [name.replace('\0', '') for _, _, name in dataset.samples]
    intersection_set = set(new_names).intersection(set(sample_names))
    print('%% of similar names: %.2f%%' % (len(intersection_set) / len(dataset)))
