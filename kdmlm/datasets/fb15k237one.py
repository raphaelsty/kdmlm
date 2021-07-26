import pathlib

from mkb import datasets as mkb_datasets
from mkb import utils

__all__ = ["Fb15k237One"]


class Fb15k237One(mkb_datasets.Dataset):
    """Fb15k237 subset with only entities mentions that are defined in the vocabulary of Bert.

    Example:
    --------

    >>> from kdmlm import datasets

    >>> dataset = datasets.Fb15k237One(1)

    >>> dataset
        Fb15k237One dataset
            Batch size  1
            Entities  3501
            Relations  212
            Shuffle  True
            Train triples  34435
            Validation triples  2129
            Test triples  2399
    """

    def __init__(self, batch_size, pre_compute=True, shuffle=True, seed=42):

        self.filename = "fb15k237one"

        path = pathlib.Path(__file__).parent.joinpath(self.filename)

        super().__init__(
            train=utils.read_csv(file_path=f"{path}/train.csv"),
            valid=utils.read_csv(file_path=f"{path}/valid.csv"),
            test=utils.read_csv(file_path=f"{path}/test.csv"),
            entities=utils.read_json(f"{path}/entities.json"),
            relations=utils.read_json(f"{path}/relations.json"),
            batch_size=batch_size,
            shuffle=shuffle,
            classification=False,
            pre_compute=pre_compute,
            num_workers=0,
            seed=seed,
        )
