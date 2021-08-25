import json
import pathlib

from torch.utils import data

from .collator import Collator
from .kd_dataset import KDDataset
from .load_dataset import LoadFromJsonFolder

__all__ = ["WikiFb15k237OneRecall", "WikiFb15k237OneTest"]


class WikiFb15k237OneTest:
    """Wikipedia sample to evaluate perplexity on sentences that contains entities that are
    part of the vocabulary of Bert..

    Example
    -------

    >>> from kdmlm import datasets
    >>> test_dataset = datasets.WikiFb15k237OneTest()

    >>> for sentence in test_dataset:
    ...     break

    >>> sentence
    "Doherty joined punk band, Jerry's Kids (band) in 1982, and later moved on to Stranglehold and the ska band The Mighty Mighty Bosstones#Early history."

    >>> test_dataset[10]
    {'sentence': 'Offer excelled at | rowing |, in particular partnering his brother Jack in the Sweep (rowing). They also took part in skiffing, being members of The Skiff Club. They won the Gentlemens Double Sculls at the Skiff Championships Regatta in 1930, 1931, 1932, 1933 and 1935.', 'entity': 'rowing'}

    """

    def __init__(self):
        with open(
            pathlib.Path(__file__).parent.joinpath("wiki_fb15k237one_test/0.json"),
            encoding="utf-8",
        ) as input_file:
            self.dataset = json.load(input_file)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for document in self.dataset:
            yield document["sentence"].replace(" |", "")

    def __getitem__(self, idx):
        return self.dataset[idx]


class WikiFb15k237OneRecall(data.DataLoader):
    """Wikipedia sample to evaluate recall on wikipedia sentences that contains entities that are
    part of the vocabulary of Bert.

    Example
    -------

    >>> from kdmlm import datasets
    >>> from torch.utils import data
    >>> from transformers import DistilBertTokenizer
    >>> from mkb import datasets as mkb_datasets

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> test_dataset = datasets.WikiFb15k237OneRecall(
    ...     batch_size = 10,
    ...     tokenizer = tokenizer,
    ...     entities = kb.entities
    ... )

    >>> for sample in test_dataset:
    ...     break

    >>> sample.keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask'])

    >>> sample["input_ids"].shape[0]
    10

    """

    def __init__(self, batch_size, tokenizer, entities, n_masks=1):
        super().__init__(
            dataset=KDDataset(
                dataset=LoadFromJsonFolder(
                    folder=pathlib.Path(__file__).parent.joinpath("wiki_fb15k237one_test/"),
                    entities=entities,
                    shuffle=False,
                ),
                tokenizer=tokenizer,
                sep="|",
                n_masks=n_masks,
            ),
            batch_size=batch_size,
            collate_fn=Collator(tokenizer=tokenizer),
        )
