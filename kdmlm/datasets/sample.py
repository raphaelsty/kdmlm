import pathlib

from .load_dataset import LoadFromFolder

__all__ = ["Sample"]


class Sample(LoadFromFolder):
    """Some sample pre-processed sentences from wikipedia.
    Entities are delimited by the separator pipe.
    Entities come from [Fb15k237](https://github.com/raphaelsty/mkb/blob/master/mkb/datasets/fb15k237.py).

    Example:
    --------

    >>> from kdmlm import datasets

    >>> dataset = datasets.Sample()

    >>> dataset[1]
    ('Lincoln is believed to have had depression (mood), |smallpox|, and |malaria|. ', 'smallpox')

    """

    def __init__(self):
        super().__init__(pathlib.Path(__file__).parent.joinpath("sentences"))
