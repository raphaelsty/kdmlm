import json
import pathlib

from torch.utils import data

from .collator import Collator
from .kd_dataset import KDDataset
from .load_dataset import LoadFromJsonFolder

__all__ = ["WikiFb15k237Recall", "WikiFb15k237Test"]


class WikiFb15k237Test:
    """Wikipedia sample to evaluate perplexity.

    Example
    -------

    >>> from kdmlm import datasets
    >>> test_dataset = datasets.WikiFb15k237Test()

    >>> for sentence in test_dataset:
    ...     break

    >>> sentence
    'Doki Doki Morning The song introduced all three members to heavy metal music; Suzuka Nakamoto commenting how she had never heard such musical heaviness before, while Yui Mizuno initially had more interest in dancing to the music rather than singing. During song production, the signature Kitsune hand gesture (similar to the sign of the horns) was formed.'

    >>> test_dataset[10]
    {'sentence': 'Deborah Weisz Deborah Weisz is an American jazz musician. She plays lead trombone for the Deborah Weisz Quintet/Trio. Deborah Weisz was born in | Chicago | to Victor and Alice Mae. She grew up in Phoenix, Arizona, where she began to play trombone at the age of ten.', 'entity': 'Chicago'}

    """

    def __init__(self):
        with open(
            pathlib.Path(__file__).parent.joinpath("wiki_fb15k237_test/0.json"),
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


class WikiFb15k237Recall(data.DataLoader):
    """Wikipedia sample to evaluate recall.

    Example
    -------

    >>> from kdmlm import datasets
    >>> from torch.utils import data
    >>> from transformers import DistilBertTokenizer
    >>> from mkb import datasets as mkb_datasets

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> test_dataset = datasets.WikiFb15k237Recall(
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
                    folder=pathlib.Path(__file__).parent.joinpath("wiki_fb15k237_test/"),
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
