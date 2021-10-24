import os

import numpy as np
import torch
import tqdm
from river import stats

__all__ = ["entity_perplexity", "sentence_perplexity", "perplexity"]


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


def entity_perplexity(dataset, model, max_step_evaluation=None, device="cpu"):
    """Evaluate perplexity over entities.

    >>> from kdmlm import datasets
    >>> from kdmlm import utils

    >>> from mkb import datasets as mkb_datasets

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> dataset = datasets.WikiFb15k237Recall(
    ...     batch_size = 10,
    ...     tokenizer = tokenizer,
    ...     entities = kb.entities
    ... )

    utils.entity_perplexity(dataset=dataset, model=model, device="cpu")

    """
    ppl = stats.Mean()

    if max_step_evaluation is None:
        max_step_evaluation = len(dataset)

    with torch.inference_mode():

        bar = tqdm.tqdm(dataset, position=0)

        for step, sample in enumerate(bar):

            if step > max_step_evaluation:
                break

            labels = sample.pop("labels")
            labels = labels[labels != -100]
            mask = sample.pop("mask")
            sample.pop("entity_ids")
            logits = model(**{key: value.to(device) for key, value in sample.items()}).logits[mask]
            for logit, label in zip(logits, labels):
                ppl.update(logit[label].item())
            bar.set_description(desc=f"PPL entities: {ppl.get():4f}")

    return ppl.get()
