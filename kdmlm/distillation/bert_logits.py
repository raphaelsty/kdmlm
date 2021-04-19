__all__ = ["BertLogits"]

import collections

import torch
import tqdm

from ..utils import distillation_index, expand_bert_logits


class BertLogits:
    """Compute bert logits for a given set of triples.

    Parameters
    ----------
        k (int): Size of the sample to distill.

    Examples
    --------

    >>> from torch.utils import data
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> from kdmlm import datasets
    >>> from kdmlm import distillation
    >>> from mkb import datasets as kb

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> dataset = data.DataLoader(
    ...    dataset = datasets.KDDataset(
    ...        dataset = datasets.Sample(),
    ...        tokenizer = tokenizer,
    ...        sep = '|',
    ...     ),
    ...     collate_fn = datasets.Collator(tokenizer=tokenizer),
    ...     batch_size = 6,
    ... )

    >>> distillation = distillation.BertLogits(
    ...     model = model,
    ...     dataset = dataset,
    ...     tokenizer = tokenizer,
    ...     entities = kb.Fb15k237(1, pre_compute=False).entities,
    ...     k = 4,
    ...     n = 1000,
    ...     device = "cpu"
    ... )

    >>> len(distillation.logits.keys())
    72

    >>> logits, index = distillation.logits[11839][0]

    >>> torch.round(logits)
    tensor([ 8.,  8.,  8.,  8., -2.,  0.,  4.,  1.])

    >>> index
    tensor([ 1166, 13596, 10039, 11737,  9133, 13765,  1840,   617])

    """

    def __init__(self, model, tokenizer, dataset, entities, k=100, n=1000, device="cuda"):
        self.k = k
        self.n = n
        self.device = device
        self.kg_entities, self.bert_entities = distillation_index(tokenizer, entities)
        self.logits = self.update(model=model, tokenizer=tokenizer, dataset=dataset)

    def update(self, model, tokenizer, dataset):
        """Compute n * batch_size distributions and store them into a dictionnary which have
        entities ids as key.

        Parameters
        ----------
            model: HuggingFace model.
            tokenizer: HuggingFace tokenizer.
            dataset: kdmlm.KdDataset.

        """
        logits = collections.defaultdict(list)
        bar = tqdm.tqdm(
            range(min(self.n // dataset.batch_size, len(dataset))),
            desc="Updating bert logits",
            position=0,
        )
        dataset = iter(dataset)

        for _ in bar:
            sample = next(dataset)
            with torch.no_grad():
                for entity, l, index in self._top_k(model=model, **sample):
                    logits[entity].append((l, index))

        return logits

    def _top_k(self, model, input_ids, attention_mask, labels, entity_ids, **kwargs):
        """Yield probabilities distribution for a given sample.

        Parameters
        ----------
            model: HuggingFace model.
            bert_entities: Torch tensor of entities index.
            kg_entities: Torch tensor of entities index.
            input_ids:  Input sentence ids.
            attention_mask: Input sample attention mask.
            labels: Input sample targets.
            entity_ids: Entities of the input sentences.

        """
        output = model(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

        logits = expand_bert_logits(
            logits=output.logits,
            labels=labels.to(self.device),
            bert_entities=self.bert_entities.to(self.device),
        )

        top_k = torch.argsort(logits, dim=1, descending=True)[:, 0 : self.k]

        random_k = torch.randint(
            low=0, high=len(self.kg_entities) - 1, size=(logits.shape[0], self.k)
        ).to(self.device)

        for i, entity in enumerate(entity_ids):

            top_k_logits = torch.index_select(logits[i], 0, top_k[i])

            random_k_logits = torch.index_select(logits[i], 0, random_k[i])

            yield entity.item(), torch.cat([top_k_logits, random_k_logits]), torch.cat(
                [top_k[i], random_k[i]]
            )