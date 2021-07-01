import json
import pathlib

__all__ = ["Fb15k237Types"]


class Fb15k237Types:
    """Fb15k237 types.

    References:
    -----------

    1. [Inductive Entity Representations from Text via Link Prediction](https://github.com/dfdazac/blp)

    Examples:
    ---------

    >>> from kdmlm import datasets

    >>> types = datasets.Fb15k237Types()

    >>> "Renault" in types
    False

    >>> types

    """

    def __init__(self):

        with open(
            pathlib.Path(__file__).parent.joinpath("fb15k237_types.json"), "r"
        ) as input_file:
            self.types = json.load(input_file)

    def __getitem__(self, idx):
        return self.types[idx]

    def __contains__(self, idx):
        return idx in self.types

    def __repr__(self) -> str:
        return str(self.types)
