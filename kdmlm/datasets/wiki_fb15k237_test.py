import json
import pathlib

from river import stats
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
    'Pentti Olavi Niemi (19021962) was a Finnish  Lutheran clergyman and politician. He was born on 9 July 1902 in tampere, and was a member of the Parliament of Finland from 1948 to 1954 and again from 1958 until his death on 7 February 1962, representing the Social Democratic Party of Finland (SDP).'

    >>> test_dataset[10]
    {'sentence': 'Lagunes District () is one of fourteen Districts of Ivory Coast of | ivorycoast |. The district is located in the southern part of the country. The capital of the district is Dabou.', 'entity': 'ivorycoast'}

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
    >>> from mkb import datasets as mkb_datasets
    >>> from transformers import DistilBertTokenizer

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> dataset = datasets.WikiFb15k237Recall(
    ...     batch_size = 10,
    ...     tokenizer = tokenizer,
    ...     entities = kb.entities
    ... )

    >>> for sample in dataset:
    ...     break

    >>> sample.keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask'])

    >>> sample["input_ids"].shape[0]
    10

    >>> tokenizer.decode(sample["input_ids"][0])
    '[CLS] panguraptor ( " pangu [ a chinese god ] plunderer " ) is a genus of coelophysidae theropod [MASK] known from fossils discovered in lower jurassic rocks of southern china. the type species and only known species is " panguraptor lufengensis ". [SEP] [PAD] [PAD] [PAD] [PAD]'

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
