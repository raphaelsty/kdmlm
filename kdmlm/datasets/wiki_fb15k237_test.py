import json
import pathlib
from xml.dom.minidom import Document

from torch.utils import data

from .collator import Collator
from .fb15k237 import Fb15k237
from .kd_dataset import KDDataset
from .load_dataset import LoadFromJsonFolder

__all__ = ["WikiFb15k237Recall", "WikiFb15k237Test"]


class WikiFb15k237Test:
    """Wikipedia sample to evaluate perplexity.

    Example
    -------

    >>> from kdmlm import datasets
    >>> from pprint import pprint as prit
    >>> test_dataset = datasets.WikiFb15k237Test()

    >>> for sentence in test_dataset:
    ...     pass

    >>> print(sentence)
    Jean Vaquette was a French Olympic weightlifting. He competed in the Weightlifting at the 1920  Men's 67.5 kg at the 1920.

    >>> test_dataset[10]
    {'sentence': 'Smithfield House is a heritage-listed villa at 8 Panda Street, Harristown, queensland, Toowoomba , Toowoomba Region, | queensland |, Australia. It was designed by architectural firm James Marks and Son and built from onwards. It was added to the queensland Heritage Register on 21 October 1992.', 'entity': 'queensland'}

    """

    def __init__(self):
        self.mentions = Fb15k237(1, pre_compute=False).mentions

        with open(
            pathlib.Path(__file__).parent.joinpath("wiki_fb15k237_test/0.json"),
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
    ... )

    >>> for sample in dataset:
    ...     break

    >>> sample.keys()
    dict_keys(['input_ids', 'labels', 'mask', 'entity_ids', 'attention_mask'])

    >>> sample["input_ids"].shape[0]
    10

    >>> tokenizer.decode(sample["input_ids"][1])
    '[CLS] the string quartet no. 2 ( deutsch catalogue 32 ) in c major was composed by [MASK] in 1812. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

    >>> tokenizer.decode(sample["labels"][1])
    '[UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] franz [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK]'

    """

    def __init__(self, batch_size, tokenizer, n_masks=1):

        kb = Fb15k237(1, pre_compute=False)

        super().__init__(
            dataset=KDDataset(
                dataset=LoadFromJsonFolder(
                    folder=pathlib.Path(__file__).parent.joinpath("wiki_fb15k237_test/"),
                    entities=kb.entities,
                    shuffle=False,
                ),
                tokenizer=tokenizer,
                sep="|",
                n_masks=n_masks,
                mentions=kb.ids_to_labels,
            ),
            batch_size=batch_size,
            collate_fn=Collator(tokenizer=tokenizer),
        )
