__all__ = ["BertLogits"]

import collections

import torch
import tqdm

from ..utils import bert_top_k, distillation_index, mapping_entities


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

    >>> kb = datasets.Fb15k237(1, pre_compute=False)

    >>> distillation = distillation.BertLogits(
    ...     model = model,
    ...     dataset = dataset,
    ...     tokenizer = tokenizer,
    ...     entities = kb.ids_to_labels,
    ...     k = 100,
    ...     n = 10000,
    ...     device = "cpu",
    ...     max_tokens = 15,
    ...     subwords_limit = 10,
    ...     average = True,
    ... )

    >>> logits, index = distillation.logits[1197][0]

    >>> e = {v: k for k, v in kb.entities.items()}

    >>> e[4688]
    'Liberia'

    >>> for i in index.tolist()[:5]:
    ...     print(e[i])
    Liberia
    Niger
    Sudan
    South Sudan
    Nigeria

    """

    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        entities,
        max_tokens,
        subwords_limit,
        k,
        n,
        device,
        entities_to_distill=None,
        average=False,
    ):

        self.k = k
        self.n = n
        self.max_tokens = max_tokens
        self.device = device
        self.average = average
        self.kb_entities, _ = distillation_index(
            tokenizer=tokenizer, entities=entities, subwords_limit=subwords_limit
        )

        self.tokens, self.order = mapping_entities(
            tokenizer=tokenizer, max_tokens=self.max_tokens, entities=entities
        )

        if entities_to_distill is not None:
            self.filter_entities = {
                e.item(): True for e in self.kb_entities if e.item() in entities_to_distill
            }
        else:
            self.filter_entities = {e.item(): True for e in self.kb_entities}

        self.order = self.order.to(self.device)

        self.logits = self.update(model=model, dataset=dataset)

    def update(self, model, dataset):
        """Compute n * batch_size distributions and store them into a dictionnary which have
        entities ids as key.

        Parameters
        ----------
            model: HuggingFace model.
            tokenizer: HuggingFace tokenizer.
            dataset: kdmlm.KdDataset.

        """
        mlm_probability = 0
        mlm_probability, dataset.dataset.mlm_probability = (
            dataset.dataset.mlm_probability,
            mlm_probability,
        )

        n_distributions = 0

        logits = collections.defaultdict(list)

        # Compute means of distributions
        if self.average:
            average_logits = {}
            frequency = collections.defaultdict(int)

        bar = tqdm.tqdm(
            range(min(self.n // dataset.batch_size, len(dataset))),
            desc=f"Updating Bert logits, {n_distributions} distributions, {len(logits)} entities.",
            position=0,
        )

        with bar:

            for sample in dataset:

                with torch.no_grad():

                    if "entity_ids" not in sample:
                        continue

                    # If not all samples have a label:

                    sum_labels = (sample["labels"] != -100).sum().item()
                    if (
                        sum_labels != sample["labels"].shape[0]
                        or sum_labels != sample["entity_ids"].shape[0]
                    ):
                        continue

                    # Wether or not to compute average distributions:
                    if self.average:
                        for entity, logit in self._compute_logits(model=model, **sample):
                            # Filter entities
                            if entity not in self.filter_entities:
                                continue
                            if entity in average_logits:
                                average_logits[entity] += logit
                            else:
                                average_logits[entity] = logit
                            n_distributions += 1
                            frequency[entity] += 1
                    else:
                        for entity, l, index in self._top_k(model=model, **sample):
                            # Filter entities
                            if entity not in self.filter_entities:
                                continue
                            logits[entity].append((l, index))
                            n_distributions += 1

                    bar.update(1)
                    n_logits = len(logits)
                    bar.set_description(
                        f"Updating Bert logits, {n_distributions} distributions, {n_logits} entities."
                    )

                    if n_distributions >= self.n:
                        break

        dataset.dataset.mlm_probability = mlm_probability
        if self.average:
            logits = self._logits_from_average(average_logits=average_logits, frequency=frequency)
        return logits

    def _compute_logits(self, model, input_ids, attention_mask, labels, entity_ids, **kwargs):
        outputs = model(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )
        logits = outputs.logits[labels != -100]
        for i, entity in enumerate(entity_ids):
            yield entity.item(), logits[i]

    def _logits_from_average(self, average_logits: dict, frequency: dict):
        """Retrieve logits distribution, i.e index, scores from average of logits."""
        logits = collections.defaultdict(list)

        for entity, logit in average_logits.items():

            logit = (logit / frequency[entity]).unsqueeze(0)

            top_k_logits, logit, top_k = bert_top_k(
                logits=logit,
                tokens=self.tokens,
                order=self.order,
                max_tokens=self.max_tokens,
                k=self.k,
                device=self.device,
            )

            random_k = torch.randint(
                low=0, high=len(self.kb_entities) - 1, size=(logit.shape[0], self.k)
            ).to(self.device)

            random_k_logits = torch.index_select(logit[0], 0, random_k[0])

            logits[entity].append(
                (
                    torch.cat([top_k_logits[0], random_k_logits]),
                    torch.cat([top_k[0], random_k[0]]),
                )
            )

        return logits

    def _top_k(self, model, input_ids, attention_mask, labels, entity_ids, **kwargs):
        """Yield probabilities distribution for a given sample.

        Parameters
        ----------
            model: HuggingFace model.
            bert_entities: Torch tensor of entities index.
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

        top_k_logits, logits, top_k = bert_top_k(
            logits=logits,
            tokens=self.tokens,
            order=self.order,
            max_tokens=self.max_tokens,
            k=self.k,
            device=self.device,
        )

        random_k = torch.randint(
            low=0, high=len(self.kb_entities) - 1, size=(logits.shape[0], self.k)
        ).to(self.device)

        for i, entity in enumerate(entity_ids):
            random_k_logits = torch.index_select(logits[i], 0, random_k[i])

            yield entity.item(), torch.cat([top_k_logits[i], random_k_logits]), torch.cat(
                [top_k[i], random_k[i]]
            )
