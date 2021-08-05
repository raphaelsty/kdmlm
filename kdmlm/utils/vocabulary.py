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

    >>> from mkb import datasets
    >>> from kdmlm import utils
    >>> from transformers import DistilBertTokenizer, DistilBertForMaskedLM

    >>> dataset = datasets.Fb15k237(1, pre_compute=False, num_workers=0)
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> len(tokenizer)
    30522

    >>> model, tokenizer = utils.expand_bert_vocabulary(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    entities = dataset.entities
    ... )

    >>> len(tokenizer)
    43225

    """
    entities_to_add = {}

    # Expand bert vocabulary for entities that are not already part of it's vocabulary:
    for e in tqdm.tqdm(entities, position=0, desc="Adding entities to Bert vocabulary."):
        e = e.lower() if lower else e
        tokenized_e = tokenizer.tokenize(e)
        if len(tokenized_e) > 1:
            entities_to_add[e] = tokenized_e

    # Expand bert vocabulary:
    tokenizer.add_tokens(list(entities_to_add.keys()))
    model.resize_token_embeddings(len(tokenizer))
    vocabulary = tokenizer.get_vocab()

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
