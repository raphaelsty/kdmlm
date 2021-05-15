import os
import random

import torch
import tqdm

__all__ = ["LoadFromFile", "LoadFromFolder", "LoadFromTorch", "LoadFromTorchFolder"]


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
        file = open(path, "r", encoding="utf-8", errors="ignore")
        return file.readlines()


class LoadFromTorch:
    """Load data from torch file.

    Arguments:
    ----------
        path (str): Path to the file storing inputs.

    """

    def __init__(self, path):
        self.dataset = self.load(path=path)
        self.len_file = self.dataset["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.dataset.items()}

    def __len__(self):
        return self.len_file

    @classmethod
    def load(cls, path):
        """Load torch file."""
        return torch.load(path)


class LoadFromFolder(LoadFromFile):
    """Load data from folder.

    Arguments:
    ----------
        path (str): Path to the folder storing sentences.

    Exemple:
    --------

    >>> import pathlib
    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

    >>> from kdmlm import datasets
    >>> from mkb import datasets as mkb_datasets

    >>> from transformers import DistilBertTokenizer

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> dataset = datasets.LoadFromFolder(folder=folder, entities=kb.entities, tokenizer=tokenizer)

    >>> dataset[2]
    ('Realizing Clay was unlikely to win the presidency, he supported General | Zachary Taylor | for the Whig nomination in the a  ', 11839)

    >>> for i in range(1000):
    ...    _ = dataset[i]

    """

    def __init__(
        self, folder, entities, tokenizer, subwords_limit=15, sep="|", shuffle=False, seed=42
    ):
        self.folder = folder
        self.list_files = os.listdir(folder)
        self.call = 0
        self.id_file = 0
        self.sep = sep
        self.entities = entities
        self.subwords_limit = subwords_limit

        self.filter = {
            idx: True
            for e, idx in tqdm.tqdm(
                entities.items(),
                position=0,
                desc=f"Filtering entities composed of > {self.subwords_limit} subwords",
            )
            if len(tokenizer.encode(e, add_special_tokens=False)) <= self.subwords_limit
        }

        if shuffle:
            random.seed(42)
            random.shuffle(self.list_files)

        super().__init__(path=os.path.join(self.folder, self.list_files[self.id_file]))

    def __getitem__(self, idx):

        if (self.call + 1) >= self.len_file:

            # We iterate over a complete file.
            self.call = 0

            # If we have been trough all the file, i.e a complete epoch:
            if (self.id_file + 1) == len(self.list_files):
                self.id_file = 0
            else:
                self.id_file += 1

            self.dataset = self.load(path=os.path.join(self.folder, self.list_files[self.id_file]))

        i = 0
        while True:
            try:
                sentence = self.dataset[self.call + i].replace("\n", "")
                entity_id = self.entities[sentence.split(self.sep)[1].strip()]
                self.call += 1
                if entity_id in self.filter:
                    return sentence, entity_id
            except:
                # Need to switch of file:
                if self.call + i >= self.len_file:
                    self.call += i
                    return self.__getitem__(idx)
                # Just an entity not correctly parsed:
                pass

            i += 1

    def __len__(self):
        return self.len_file * len(self.list_files)


class LoadFromTorchFolder(LoadFromTorch):
    """Load torch serialized samples from folder."""

    def __init__(self, folder, shuffle=False, seed=42):
        self.folder = folder
        self.list_files = os.listdir(folder)
        self.call = 0
        self.id_file = 0

        if shuffle:
            random.seed(42)
            random.shuffle(self.list_files)

        super().__init__(path=os.path.join(self.folder, self.list_files[self.id_file]))

    def __getitem__(self, idx):
        if (self.call + 1) >= self.len_file:
            # We iterate over a complete file.
            self.call = 0
            # If we have been trough all the file, i.e a complete epoch:
            if (self.id_file + 1) == len(self.list_files):
                self.id_file = 0
            else:
                self.id_file += 1
            self.dataset = self.load(path=os.path.join(self.folder, self.list_files[self.id_file]))

        self.call += 1
        return super().__getitem__(idx=self.call)

    def __len__(self):
        return self.len_file * len(self.list_files)

    @staticmethod
    def collate_fn(data):
        output = {
            "input_ids": [],
            "labels": [],
            "mask": [],
            "entity_ids": [],
            "attention_mask": [],
        }
        for x in data:
            for key, value in x.items():
                output[key].append(value)

        for key, value in output.items():
            output[key] = torch.stack(value, dim=0)
        return output
