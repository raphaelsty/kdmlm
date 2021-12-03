import json
import pathlib

from mkb import datasets as mkb_datasets
from transformers import DistilBertTokenizer

__all__ = ["Fb15k237"]


class Fb15k237(mkb_datasets.Dataset):
    """Fb15k237 but spaces in entities names are deleted so it can be added to Bert vocabulary
    as a full token. I made this knowledge base to add all entities to bert vocabulary while
    avoiding issues with the speedness of the tokenizer. The trick is to remove spaces in entities
    names.


    Example
    -------

    >>> from kdmlm import datasets
    >>> from kdmlm import utils

    >>> from transformers import DistilBertTokenizer, DistilBertForMaskedLM

    >>> kb = datasets.Fb15k237(1, pre_compute=False)

    >>> len(kb.mentions) == len(kb.entities)
    True

    >>> {v: k for k, v in kb.entities.items()}[0]
    'Dominican Republic'

    >>> kb.label(e = 0)
    'dominican'

    >>> kb.label(e = "Dominican Republic")
    'dominican'

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> model, tokenizer = utils.expand_bert_vocabulary(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    entities = kb.mentions.values(),
    ...    original_entities = kb.original_mentions,
    ... )

    >>> n = 0

    >>> for e in kb.mentions.values():
    ...     if len(tokenizer.tokenize(e)) == 1:
    ...         n += 1

    >>> n
    14535

    >>> kb.ids_to_labels

    References
    ----------
    1. (tokenizer is slow after adding new tokens #615)[https://github.com/huggingface/tokenizers/issues/615]

    """

    def __init__(
        self,
        batch_size,
        tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        classification=False,
        shuffle=True,
        pre_compute=True,
        num_workers=0,
        seed=42,
    ):

        freebase = mkb_datasets.Fb15k237(batch_size=1, pre_compute=False, num_workers=0)

        path_mentions = pathlib.Path(__file__).parent.joinpath(
            "mapping_entities_mentions_freebase.json"
        )

        with open(path_mentions, "r") as input_mentions:
            mentions = json.load(input_mentions)

        replacements = [
            (" ", ""),
            ("-", ""),
            (".", ""),
            ("'", ""),
            ("/", ""),
            (":", ""),
            ("_", ""),
            ('"', ""),
            ("!", ""),
            (",", ""),
            ("+", ""),
            ("&", ""),
            ("-", ""),
            ("?", ""),
            ("(", ""),
            (")", ""),
            ("–", ""),
            ("*", ""),
            ("’", ""),
            ("é", "e"),
            ("·", ""),
            ("ó", "o"),
            ("ō", "o"),
            ("ô", "o"),
            ("ã", "a"),
            (" ", ""),
        ]

        self.mentions = {}
        self.original_mentions = {}

        for entity, mention in mentions.items():

            updated_mention = mention.lower()

            if len(tokenizer.tokenize(mention)) == 1:
                self.mentions[entity] = updated_mention
            else:
                for target, replace in replacements:
                    updated_mention = updated_mention.replace(target, replace)

            self.mentions[entity] = updated_mention
            self.original_mentions[updated_mention] = mention

        for entity in freebase.entities:

            if entity in self.mentions:
                continue

            upated_mention = entity.lower()

            if len(tokenizer.tokenize(mention)) == 1:
                self.mentions[entity] = upated_mention
            else:
                for target, replace in replacements:
                    upated_mention = upated_mention.replace(target, replace)

                self.mentions[entity] = upated_mention
                self.original_mentions[upated_mention] = mention

        self.id_to_mention = {id_e: self.mentions[e] for e, id_e in freebase.entities.items()}

        super().__init__(
            batch_size=batch_size,
            train=freebase.train,
            valid=freebase.valid,
            test=freebase.test,
            entities=freebase.entities,
            relations=freebase.relations,
            classification=classification,
            shuffle=shuffle,
            pre_compute=pre_compute,
            num_workers=num_workers,
            seed=seed,
        )

    @property
    def ids_to_labels(self):
        return {id_e: self.label(id_e) for _, id_e in self.entities.items()}

    def label(self, e):
        """Returns the most frequent mention for an entity."""
        if isinstance(e, int):
            return self.id_to_mention[e]
        else:
            return self.mentions[e]
