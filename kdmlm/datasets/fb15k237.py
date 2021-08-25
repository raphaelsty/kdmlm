from mkb import datasets as mkb_datasets

from .fb15k237one import Fb15k237One

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
    >>> one_token_entities = datasets.Fb15k237One(1, pre_compute=False).entities

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> model, tokenizer = utils.expand_bert_vocabulary(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    entities = {k: v for k, v in kb.entities.items() if k not in one_token_entities},
    ... )

    >>> n = 0

    >>> for e in kb.entities:
    ...     if len(tokenizer.tokenize(e)) == 1 and e not in one_token_entities:
    ...         n += 1

    >>> n
    11014

    Number of entities that are not part of Bert vocabulary, i.e entities with special caracters:
    >>> len(kb.entities) - (n + len(one_token_entities))
    17

    >>> kb.mapping

    References
    ----------
    1. (tokenizer is slow after adding new tokens #615)[https://github.com/huggingface/tokenizers/issues/615]

    """

    def __init__(
        self,
        batch_size,
        classification=False,
        shuffle=True,
        pre_compute=True,
        num_workers=0,
        seed=42,
    ):

        freebase = mkb_datasets.Fb15k237(batch_size=1, pre_compute=False, num_workers=0)
        freebase_one = Fb15k237One(1, pre_compute=False, shuffle=False)

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
        ]

        new_entities = {}
        for e, id in freebase.entities.items():
            if e in freebase_one.entities:
                new_entities[e] = id
            else:
                e = e.lower()
                for target, replace in replacements:
                    e = e.replace(target, replace)
                new_entities[e] = id

        super().__init__(
            batch_size=batch_size,
            train=freebase.train,
            valid=freebase.valid,
            test=freebase.test,
            entities=new_entities,
            relations=freebase.relations,
            classification=classification,
            shuffle=shuffle,
            pre_compute=pre_compute,
            num_workers=num_workers,
            seed=seed,
        )

    @property
    def mapping(self):
        """Mapping between mkb.Fb15k237 entities labels and kdmlm.Fb15k237 entities labels."""
        freebase_entities = mkb_datasets.Fb15k237(
            batch_size=1, pre_compute=False, num_workers=0
        ).entities
        freebase_entities = {id: label for label, id in freebase_entities.items()}
        return {freebase_entities[id]: label for label, id in self.entities.items()}
