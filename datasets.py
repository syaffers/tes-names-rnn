import os
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class TESNamesDataset(Dataset):
    ''' The Elder Scrolls Names dataset class.

    The Elder Scrolls Names dataset is a dataset of first names of all 10
    major races in Tamriel. The dataset contains male and female first names
    of different lengths and characters organized into folders of races and
    files of genders.
    '''
    def __init__(self, data_root, charset, max_length):
        ''' Initializes the Elder Scrolls dataset.

        The initialization appends a terminating character, \0, and therfore
        the passed charset argument should not contain the terminating
        character.

        Parameters
        ----------
        data_root: str
            Absolute path to the root folder of the dataset.
        charset: str
            String of all characters expected to be present in the names.
        max_length: str
            The maximum number of characters in a name to be used for
            zero-padding or truncation of names when preprocessing.
        '''
        self.data_root = data_root
        self.charset = charset + '\0'
        self.max_length = max_length
        self.race_codec = LabelEncoder()
        self.gender_codec = LabelEncoder()
        self.char_codec = LabelEncoder()
        self.samples = []
        self._init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        race, gender, name = self.samples[idx]
        return self.one_hot_sample(race, gender, name)

    def _init_dataset(self):
        ''' Dataset initialization subroutine.

        Goes through all the folders in the root directory of the dataset
        and reads all the files in the subfolders and appends tuples of
        race, gender and name into the `self.samples` list.

        The label encoder for the races, genders and characters are also
        initialized here.
        '''
        races = set()
        genders = set()

        for race in os.listdir(self.data_root):
            race_folder = os.path.join(self.data_root, race)
            races.add(race)

            for gender in os.listdir(race_folder):
                gender_filepath = os.path.join(race_folder, gender)
                genders.add(gender)

                with open(gender_filepath, 'r') as gender_file:
                    for name in gender_file.read().splitlines():
                        if len(name) > self.max_length:
                            name = name[:self.max_length-1] + '\0'
                        else:
                            name = name + '\0' * (self.max_length - len(name))
                        self.samples.append((race, gender, name))

        self.race_codec.fit(list(races))
        self.gender_codec.fit(list(genders))
        self.char_codec.fit(list(self.charset))

    def to_one_hot(self, codec, values):
        ''' Encodes a list of nominal values into a one-hot tensor.

        Parameters
        ----------
        codec: sklearn.preprocessing.LabelEncoder
            Scikit-learn label encoder for the list of values.
        values: list of str
            List of values to be converted into numbers.
        '''
        values_idx = codec.transform(values)
        return torch.eye(len(codec.classes_))[values_idx]

    def one_hot_sample(self, race, gender, name):
        ''' Converts a single sample into its one-hot counterpart.

        Calls the `to_one_hot` function for each of the value in a sample:
        race, gender, and name. The race and gender gets converted into
        a 1xR tensor, and 1xG tensor, respectively, where R is the number of
        races in the dataset and G is the number of genders in the dataset.

        The name gets converted into a tensor of 1xMxC where M is the maximum
        length of the names (`self.max_length`) and C is the length of the
        character set (after adding the terminationg, \0, character).

        Parameters
        ----------
        race: str
            The race of the sample.
        gender: str
            The gender of the sample.
        name: str
            The name of the sample.
        '''
        t_race = self.to_one_hot(self.race_codec, [race])
        t_gender = self.to_one_hot(self.gender_codec, [gender])
        t_name = self.to_one_hot(self.char_codec, list(name))
        return t_race, t_gender, t_name


if __name__ == '__main__':
    import string
    from torch.utils.data import DataLoader

    data_root = '/home/syafiq/Data/tes-names/'
    charset = string.ascii_letters + '\'- '
    max_length = 30
    dataset = TESNamesDataset(data_root, charset, max_length)
    print(dataset[100])

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(next(iter(dataloader)))
