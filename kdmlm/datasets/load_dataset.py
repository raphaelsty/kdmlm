import os
import random

__all__ = ["LoadFromFolder", "LoadFromFile", "LoadFromStream"]


class LoadFromFile:
    """Load data from file.

    Arguments:
    ----------
        path (str): Path to the file storing sentences.

    Example:
    --------
    """

    def __init__(self, path):
        self.dataset = self.load(path=path)
        self.len_file = sum(1 for line in open(path))

    def __getitem__(self, idx):
        return self.dataset[idx].replace("\n", "")

    def __len__(self):
        return self.len_file

    @classmethod
    def load(cls, path):
        """Load txt file."""
        file = open(path, "r")
        return file.readlines()


class LoadFromFolder(LoadFromFile):
    """Load data from folder.

    Arguments:
    ----------
        path (str): Path to the folder storing sentences.

    Exemple:
    --------

    >>> from kdmlm import datasets
    >>> from mkb import datasets as mkb_datasets

    >>> import pathlib
    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    >>> dataset = datasets.LoadFromFolder(folder=folder, entities=kb.entities)

    >>> dataset[2]
    ('Realizing Clay was unlikely to win the presidency, he supported General | Zachary Taylor | for the Whig nomination in the a  ', 11839)

    >>> for i in range(1000):
    ...    _ = dataset[i]

    """

    def __init__(self, folder, entities, sep="|", shuffle=False, seed=42):
        self.folder = folder
        self.list_files = os.listdir(folder)
        self.call = 0
        self.id_file = 0
        self.sep = sep
        self.entities = entities

        if shuffle:
            random.seed(42)
            random.shuffle(self.list_files)

        super().__init__(path=os.path.join(self.folder, self.list_files[self.id_file]))

    def __getitem__(self, idx):

        if (self.call + 1) == self.len_file:

            # We iterate over a complete file.
            self.call = 0

            # If we have been trough all the file, i.e a complete epoch:
            if (self.id_file + 1) == len(self.list_files):
                self.id_file = 0
            else:
                self.id_file += 1

            self.dataset = self.load(path=os.path.join(self.folder, self.list_files[self.id_file]))

        for i in range(100):
            try:
                sentence = self.dataset[self.call + i].replace("\n", "")
                entity_id = self.entities[sentence.split(self.sep)[1].strip()]
                self.call += 1
                return sentence, entity_id
            except:
                continue

    def __len__(self):
        return self.len_file * len(self.list_files)


class LoadFromStream:
    """Load data from Stream.

    Arguments:
    ----------
        x (list): Input sentences.

    Example:
    --------
    """

    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass
