import pathlib

from .load_dataset import LoadFromTorchFolder

__all__ = ["TorchSample"]


class TorchSample(LoadFromTorchFolder):
    """Some sample pre-processed sentences from wikipedia.

    Example:
    --------

    >>> from kdmlm import datasets

    >>> dataset = datasets.TorchSample()

    >>> dataset[0].keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask'])

    >>> from torch.utils.data import DataLoader

    >>> dataset = DataLoader(dataset = dataset, collate_fn = dataset.collate_fn, batch_size = 2)

    >>> for data in dataset:
    ...     break

    >>> data.keys()
     dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask', 'mode', 'triple'])

    """

    def __init__(self):
        super().__init__(
            pathlib.Path(__file__).parent.joinpath("torch_sentences"),
        )
