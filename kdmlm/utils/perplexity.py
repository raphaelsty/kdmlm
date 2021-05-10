import os

import numpy as np
import torch

__all__ = ["perplexity"]


def perplexity(model, tokenizer, sentence):
    """Computes perplexity.


    Examples
    --------

    >>> from kdmlm import utils

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> utils.perplexity(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    sentence = 'Barack Obama is the president.'
    ... )
    3.68901411315264

    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    tensor_input = tokenizer.encode(sentence, return_tensors="pt")
    repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.no_grad():
        loss = model(input_ids=masked_input, labels=labels).loss.item()
    return np.exp(loss)
