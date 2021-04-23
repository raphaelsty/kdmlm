__all__ = ["BertLogits"]

import collections

import torch
import tqdm

from ..utils import bert_top_k, distillation_index, expand_bert_logits, mapping_entities


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

    >>> kb = kb.Fb15k237(1, pre_compute=False)

    >>> distillation = distillation.BertLogits(
    ...     model = model,
    ...     dataset = dataset,
    ...     tokenizer = tokenizer,
    ...     entities = kb.entities,
    ...     k = 4,
    ...     n = 1000,
    ...     device = "cpu"
    ... )

    >>> len(distillation.logits.keys())
    72

    >>> logits, index = distillation.logits[11839][0]

    >>> torch.round(logits)
    tensor([ 7.,  7.,  7.,  7.,  0., -6., -5., -5.])

    >>> index
    tensor([ 9481,   753,  4931,  2293,  9133, 13765,  1840,   617])

    >>> e = {v: k for k, v in kb.entities.items()}

    >>> e[11839]
    'Zachary Taylor'

    >>> for i in index:
    ...     print(e[i.item()])
    John Abraham
    Richard Benjamin
    Henry James
    William James
    Texas A&M University
    Liberal Party of Australia
    Anthony Burgess
    The Devil's Double

    """

    def __init__(
        self, model, tokenizer, dataset, entities, k=100, n=1000, max_tokens=15, device="cuda"
    ):
        self.k = k
        self.n = n
        self.max_tokens = max_tokens
        self.device = device
        self.kg_entities, _ = distillation_index(tokenizer, entities)

        self.tokens, self.order = mapping_entities(
            tokenizer=tokenizer, max_tokens=self.max_tokens, entities=entities
        )

        self.order = self.order.to(self.device)

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
        outputs = model(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

        logits = outputs.logits[labels != -100]

        top_k_logits, top_k = bert_top_k(
            logits=logits,
            tokens=self.tokens,
            order=self.order,
            max_tokens=self.max_tokens,
            k=self.k,
            device=self.device,
        )

        random_k = torch.randint(
            low=0, high=len(self.kg_entities) - 1, size=(logits.shape[0], self.k)
        ).to(self.device)

        for i, entity in enumerate(entity_ids):

            random_k_logits = torch.index_select(logits[i], 0, random_k[i])

            yield entity.item(), torch.cat([top_k_logits[i], random_k_logits]), torch.cat(
                [top_k[i], random_k[i]]
            )
