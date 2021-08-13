import os
from audioop import reverse

import torch
import tqdm

__all__ = ["expand_bert_vocabulary"]


def expand_bert_vocabulary(model, tokenizer, entities, lower=True):
    """Expand bert vocabulary by adding entities to bert embeddings. Entities weights are
    initialized by computing the mean of the embeddings of the tokens of the entities. This
    function is made for models which use word piece tokenizer.

    Parameters
    ----------

        model: Bert like model.
        tokenizer: Tokenizer for bert like model.
        entities: Dict or list of entities to add into the vocabulary of Bert.
        lower: Do wether or not uncase entities.

    Example
    -------

    >>> from mkb import datasets as mkb_datasets
    >>> from kdmlm import datasets
    >>> from kdmlm import utils
    >>> from transformers import DistilBertTokenizer, DistilBertForMaskedLM

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False, num_workers=0)
    >>> do_not_add_entities = datasets.Fb15k237One(1).entities
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> len(tokenizer)
    30522

    >>> model, tokenizer = utils.expand_bert_vocabulary(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    entities = {k: v for k, v in kb.entities.items() if k not in do_not_add_entities}
    ... )

    >>> len(tokenizer)
    41071

    assert 30522 + len({k: v for k, v in kb.entities.items() if k not in do_not_add_entities}) == 41071

    """
    entities_to_add = {}

    vocabulary = tokenizer.get_vocab()

    # Expand bert vocabulary for entities that are not already part of it's vocabulary:
    for e in tqdm.tqdm(entities, position=0, desc="Adding entities to Bert vocabulary."):
        e = e.lower() if lower else e

        if e not in vocabulary:
            entities_to_add[e] = tokenizer.tokenize(e)

    # Expand bert vocabulary:
    tokenizer.save_pretrained(".")

    with open("vocab.txt", "a") as file:
        for i, e in enumerate(entities_to_add):
            if i >= len(entities_to_add):
                file.write(f"{e}")
                continue
            file.write(f"{e}\n")

    tokenizer = tokenizer.from_pretrained(".")
    vocabulary = tokenizer.get_vocab()
    model.resize_token_embeddings(len(tokenizer))

    # Initialize smart weights for entities thar are now part of the vocabulary of Bert.
    # The embeddings of Toulouse University will be equal to the mean of the embeddings of
    # Toulouse and University.
    for layer in model.parameters():

        if layer.shape[0] == len(tokenizer):

            for e, tokens in entities_to_add.items():

                with torch.no_grad():

                    mean_vector = torch.zeros(
                        layer.shape[1] if len(layer.shape) == 2 else 1, requires_grad=True
                    )

                    for token in tokens:
                        # Part of bert vocabulary
                        mean_vector += layer[vocabulary[token]].detach().clone()

                    layer[vocabulary[e]] = mean_vector / len(tokens)

    return model, tokenizer
