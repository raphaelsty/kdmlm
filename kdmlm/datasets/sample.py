import pathlib

from mkb import datasets as mkb_datasets
from transformers import BertTokenizer

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
    ('Realizing Clay was unlikely to win the presidency, he supported General | Zachary Taylor | for the Whig nomination in the a  ', 11839)

    >>> from transformers import BertTokenizer
    >>> from mkb import datasets
    >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    >>> entities = {value: key for key, value in datasets.Fb15k237(1).entities.items()}



    """

    def __init__(self):
        kb = mkb_datasets.Fb15k237(1, pre_compute=False)
        super().__init__(
            pathlib.Path(__file__).parent.joinpath("sentences"),
            entities=kb.entities,
            tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
            subwords_limit=15,
        )
