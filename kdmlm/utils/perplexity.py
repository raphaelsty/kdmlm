import os

import numpy as np
import torch

__all__ = ["sentence_perplexity", "perplexity"]


def sentence_perplexity(model, tokenizer, sentence, device="cpu"):
    """Computes perplexity.

    Example
    -------

    >>> from kdmlm import utils

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> utils.sentence_perplexity(
    ...    model = model,
    ...    tokenizer = tokenizer,
    ...    sentence = 'Barack Obama is the president.',
    ... )
    3.68901411315264

    """
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    return perplexity(
        model=model, input_ids=input_ids, mask_token_id=tokenizer.mask_token_id, device=device
    )


def perplexity(model, input_ids, mask_token_id, device="cpu"):
    """

    Example
    -------

    >>> from kdmlm import utils

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> utils.perplexity(
    ...    model = model,
    ...    input_ids = tokenizer.encode('Barack Obama is the president.', return_tensors="pt"),
    ...    mask_token_id = tokenizer.mask_token_id,
    ... )
    3.68901411315264

    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    repeat_input = input_ids.repeat(input_ids.size(-1) - 2, 1)
    mask = torch.ones(input_ids.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)
    labels = repeat_input.masked_fill(masked_input != mask_token_id, -100)
    with torch.no_grad():
        loss = model(input_ids=masked_input.to(device), labels=labels.to(device)).loss.item()
    return np.exp(loss)
