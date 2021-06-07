import pathlib

from .load_dataset import LoadFromTorchFolder

__all__ = ["TriplesTorchSample"]


class TriplesTorchSample(LoadFromTorchFolder):
    """Some sample pre-processed sentences from wikipedia.
    Entities are delimited by the separator pipe.
    Entities come from [Fb15k237](https://github.com/raphaelsty/mkb/blob/master/mkb/datasets/fb15k237.py).

    Example:
    --------

    >>> from kdmlm import datasets

    >>> dataset = datasets.TriplesTorchSample()

    >>> dataset[0].keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'mode', 'triple', 'attention_mask'])

    >>> from torch.utils.data import DataLoader

    >>> dataset = DataLoader(dataset = dataset, collate_fn = dataset.collate_fn, batch_size = 2)

    >>> for data in dataset:
    ...     break

    >>> data.keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask', 'mode', 'triple'])

    >>> data["mode"][0]
    'tail-batch'

    >>> data["triple"][0]
    (6862, 15, 9933)

    """

    def __init__(self):
        super().__init__(
            pathlib.Path(__file__).parent.joinpath("triples_torch_sentence"),
        )
