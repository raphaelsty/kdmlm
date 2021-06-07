import itertools
import os

import torch
import tqdm

from ..datasets import Collator, LoadFromFile

__all__ = ["WikiTriples"]


class WikiTriples:
    """

    Example:
    --------

    >>> from kdmlm import utils

    >>> from mkb import datasets

    >>> from transformers import DistilBertTokenizer

    >>> kb = datasets.Fb15k237(1, pre_compute = False)
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> wiki_triples = utils.WikiTriples(
    ...    folder = "./../../data/wiki_fb15k237_512",
    ...    train_triples = kb.train,
    ...    entities = kb.entities,
    ...    tokenizer = tokenizer,
    ... )

    >>> wiki_triples.process(path = "test")

    """

    def __init__(self, folder, train_triples, entities, tokenizer, sep="|"):
        self.folder = folder
        self.id_file = 0
        self.list_files = os.listdir(self.folder)
        self.sep = sep

        self.tokenizer = tokenizer

        self.triples = {}
        for h, r, t in train_triples:
            self.triples[(h, t)] = (h, r, t)

        self.entities = entities
        self.collator = Collator(tokenizer=tokenizer)

    def process(self, path):
        n = 0
        i = 0
        samples = []
        for file in tqdm.tqdm(self.list_files):
            for sentence in LoadFromFile(os.path.join(self.folder, file)):
                try:
                    x = self.extract_entities(sentence)
                except KeyError:
                    continue

                if x is None:
                    continue

                data = self.get_mask_labels_ids(
                    sentence=self.tokenizer.tokenize(sentence),
                    input_ids=self.tokenizer.encode(sentence),
                    n_masks=1,
                )
                data.update(x)
                samples.append(data)
                n += 1
                if n % 10000 == 0:
                    torch.save(self.collate_fn(samples), os.path.join(path, f"{i}.torch"))
                    i += 1
                    samples = []

    def extract_entities(self, content):
        """Parse entities between sep."""
        entities = []

        for i, entity in enumerate(content.split(self.sep)):
            if not i % 2 == 0:
                e = self.entities[entity.strip()]
                entities.append(e)
                if i == 1:
                    first_entity = e

        entities = list(itertools.chain(*[[(h, t) for h in entities if h != t] for t in entities]))
        for h, t in entities:
            if h == first_entity or t == first_entity:
                if (h, t) in self.triples:
                    return {
                        "triple": self.triples[(h, t)],
                        "mode": "head-batch" if h == first_entity else "tail-batch",
                        "entity_ids": torch.tensor([first_entity]),
                    }
        return None

    def get_mask_labels_ids(self, sentence, input_ids, n_masks=None):
        mask, labels = [], []
        stop, entities = False, False
        stop_label, label = False, False
        ids = []
        n_masked = 0

        sentence.insert(0, self.tokenizer.cls_token)
        sentence.append(self.tokenizer.sep_token)

        for token, input_id in zip(sentence, input_ids):

            if self.sep in token:

                if not entities:
                    # Begining of an entity.
                    entities, label = True, True
                else:
                    # Ending of an entity.
                    # We will stop masking entities.
                    entities, stop = False, True

                    if n_masks is not None:
                        if n_masked < n_masks:
                            for _ in range(n_masks - n_masked):
                                ids.append(self.tokenizer.mask_token_id)
                                mask.append(True)
                                labels.append(-100)
            else:

                if stop:
                    # First entity already met.
                    entities = False

                if stop_label:
                    # First element of first entity already met.
                    label = False

                if entities:
                    n_masked += 1

                if n_masks is not None:
                    if entities and n_masked > n_masks:
                        continue

                ids.append(input_id)
                mask.append(entities)

                if label:
                    # Eval loss.
                    labels.append(input_id)
                else:
                    # Do not eval loss.
                    labels.append(-100)

                # Eval loss only for the first element of the first entity met.
                if label:
                    stop_label = True

        return {"mask": mask, "labels": labels, "input_ids": ids}

    def collate_fn(self, data):
        return self.collator(data)
