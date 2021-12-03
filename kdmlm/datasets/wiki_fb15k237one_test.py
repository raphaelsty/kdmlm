import json
import pathlib

from torch.utils import data

from .collator import Collator
from .fb15k237 import Fb15k237
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
    'Bijan Allipour (born 27 March 1949, Masjed Soleyman, iran ) is an iranian business executive and upstream oil and gas expert. He is an advisor to the Oil ministry (Bijan Namdar Zangeneh) in development projects.'

    >>> test_dataset[10]
    {'sentence': 'Victor Denton War Memorial is a heritage-listed memorial at Nobby Cemetery, Nobby, queensland, | queensland |, Australia. It was made in 1915 by Bruce Brothers. It was added to the queensland Heritage Register on 21 October 1992.', 'entity': 'queensland'}

    """

    def __init__(self):
        self.mentions = Fb15k237(1, pre_compute=False).mentions

        with open(
            pathlib.Path(__file__).parent.joinpath("wiki_fb15k237one_test/0.json"),
            encoding="utf-8",
        ) as input_file:
            self.dataset = json.load(input_file)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for document in self.dataset:
            sentence = document["sentence"]
            try:
                for i, entity in enumerate(sentence.split("|")):
                    if (i + 1) % 2 == 0:
                        entity = entity.strip()
                        sentence = sentence.replace(entity, self.mentions[entity])
            except:
                pass
            yield sentence.replace(" |", "")

    def __getitem__(self, idx):
        document = self.dataset[idx]
        try:
            for i, entity in enumerate(document["sentence"].split("|")):
                if (i + 1) % 2 == 0:
                    entity = entity.strip()
                    document["sentence"] = document["sentence"].replace(
                        entity, self.mentions[entity]
                    )
                    document["entity"] = self.mentions[entity]
        except:
            pass
        return document


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
    ... )

    >>> for sample in test_dataset:
    ...     break

    >>> sample.keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask'])

    """

    def __init__(self, batch_size, tokenizer, n_masks=1):

        kb = Fb15k237(1, pre_compute=False)

        super().__init__(
            dataset=KDDataset(
                dataset=LoadFromJsonFolder(
                    folder=pathlib.Path(__file__).parent.joinpath("wiki_fb15k237one_test/"),
                    entities=kb.entities,
                    shuffle=False,
                ),
                mentions=kb.ids_to_labels,
                tokenizer=tokenizer,
                sep="|",
                n_masks=n_masks,
            ),
            batch_size=batch_size,
            collate_fn=Collator(tokenizer=tokenizer),
        )
