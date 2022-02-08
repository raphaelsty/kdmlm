import json

import torch
import tqdm

__all__ = ["expand_bert_vocabulary"]


def expand_bert_vocabulary(model, tokenizer, entities, original_entities):
    """Expand Transformers vocabulary by adding entities to bert embeddings. Entities weights are
    initialized by computing the mean of the embeddings of the tokens of the entities. This
    function is made for models which use word piece tokenizer. Also compatible with Roberta
    which has a different behaviour.

    Parameters
    ----------

        model: Bert like model.
        tokenizer: Tokenizer for bert like model.
        entities: Dict or list of entities to add into the vocabulary of Bert.
        original_entity: Mapping between entities ids and original entities labels.
        lower: Do wether or not uncase entities.

    Example
    -------

    >>> from kdmlm import datasets
    >>> from kdmlm import utils

    >>> from transformers import DistilBertTokenizer, DistilBertForMaskedLM
    >>> from transformers import FillMaskPipeline

    >>> kb = datasets.Fb15k237(1, pre_compute=False, num_workers=0)

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> pipeline = FillMaskPipeline(model = model, tokenizer = tokenizer, framework = "pt")
    >>> y_pred = pipeline(
    ...    "I have studied at [MASK].",
    ...    targets = [x for x in list(kb.mentions.values()) if len(tokenizer.tokenize(x)) == 1],
    ...    top_k = 10
    ... )

    >>> for i, top in enumerate(y_pred):
    ...     if i > 10:
    ...         break
    ...     else:
    ...         print(top["sequence"])
    i have studied at oxford.
    i have studied at ucla.
    i have studied at princeton.
    i have studied at university.
    i have studied at stanford.
    i have studied at cambridge.
    i have studied at byu.
    i have studied at berkeley.
    i have studied at columbia.
    i have studied at harrow.

    >>> len(tokenizer)
    30522

    >>> n = 0
    >>> for e in kb.mentions.values():
    ...     if len(tokenizer.tokenize(e)) == 1:
    ...         n += 1

    >>> n
    2208

    >>> len(kb.mentions.values())
    14541

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
    14531

    >>> pipeline = FillMaskPipeline(model = model, tokenizer = tokenizer, framework = "pt")
    >>> y_pred = pipeline("I have studied at [MASK].", targets=list(kb.mentions.values()), top_k=10)

    >>> for i, top in enumerate(y_pred):
    ...     if i > 10:
    ...         break
    ...     else:
    ...         print(top["sequence"])
    i have studied at oxford.
    i have studied at harvarduniversity.
    i have studied at ucla.
    i have studied at princeton.
    i have studied at yaleuniversity.
    i have studied at princetonuniversity.
    i have studied at university.
    i have studied at stanforduniversity.
    i have studied at harvardcollege.
    i have studied at stanford.

    References
    ----------
    1. (tokenizer is slow after adding new tokens #615)[https://github.com/huggingface/tokenizers/issues/615]

    """
    from transformers import RobertaForMaskedLM

    entities_to_add = {}
    vocabulary = tokenizer.get_vocab()

    # Expand bert vocabulary for entities that are not already part of it's vocabulary:
    for e in tqdm.tqdm(entities, position=0, desc="Adding entities to Bert vocabulary."):
        if len(tokenizer.tokenize(e)) > 1:
            e_tkzn = tokenizer.tokenize(original_entities[e])
            if isinstance(model, RobertaForMaskedLM):
                e = f"Ġ{e}"
            entities_to_add[e] = e_tkzn

    tokenizer.save_pretrained(".")

    with open("vocab.txt", "a") as tokenizer_vocab:
        for token in entities_to_add:
            tokenizer_vocab.write(f"{token}\n")

    if isinstance(model, RobertaForMaskedLM):
        with open("vocab.json", "r") as input_vocab:
            vocabulary = json.load(input_vocab)

        for e in tqdm.tqdm(
            entities_to_add, position=0, desc="Roberta vocab.json", total=len(entities_to_add)
        ):
            size_vocabulary = max(list(vocabulary.values()))
            if e not in vocabulary:
                vocabulary[e] = size_vocabulary + 1

        with open("vocab.json", "w") as output_vocab:
            json.dump(vocabulary, output_vocab, indent=4, ensure_ascii=False)

        with open("merges.txt", "a", encoding="utf-8") as tokenizer_vocab:
            for token in entities_to_add:
                tokenizer_vocab.write(f"Ġ{token}\n")

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
                        mean_vector += layer[vocabulary[token]].detach().clone()

                    layer[vocabulary[e]] = mean_vector / len(tokens)

    return model, tokenizer
