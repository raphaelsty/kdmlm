import os

__all__ = ["LoadFromFolder", "LoadFromFile", "LoadFromStream"]


class LoadFromFolder:
    """Load data from folder.

    Arguments:
    ----------
        path (str): Path to the folder storing sentences.

    Exemple:
    --------

    >>> from kdmlm import datasets

    >>> import pathlib
    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

    >>> dataset = datasets.LoadFromFolder(folder=folder)

    >>> dataset[2]
    'At the entrance to the Gerasimov Institute of Cinematography in |Moscow|, there is a monument that includes statues of Tarkovsky, Gennady Shpalikov and Vasily Shukshin.'

    >>> for i in range(1000):
    ...    _ = dataset[i]

    """

    def __init__(self, folder):
        self.folder = folder
        self.list_files = os.listdir(folder)

        self.call = 0
        self.id_file = 0

        file_path = os.path.join(self.folder, self.list_files[self.id_file])
        file = open(file_path, "r")

        self.dataset = file.readlines()
        self.len_file = sum(1 for line in open(file_path))

    def __getitem__(self, idx):

        if (self.call + 1) == self.len_file:

            # We iterate over a complete file.
            self.call = 0

            # If we have been trough all the file, i.e a complete epoch:
            if (self.id_file + 1) == len(self.list_files):
                self.id_file = 0
            else:
                self.id_file += 1

            file_path = os.path.join(self.folder, self.list_files[self.id_file])
            file = open(file_path, "r")
            self.dataset = file.readlines()

        sentence = self.dataset[self.call].replace("\n", "")
        self.call += 1
        return sentence

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


class LoadFromFile:
    """Load data from file.

    Arguments:
    ----------
        path (str): Path to the file storing sentences.

    Example:
    --------
    """

    def __init__(self, path):
        pass

    def __getitem__(self, idx):
        pass
