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
    ...     device = "cpu",
    ...     max_tokens = 15,
    ...     subwords_limit = 10,
    ... )

    >>> len(distillation.logits.keys())
    72

    >>> logits, index = distillation.logits[1197][0]

    >>> torch.round(logits)
    tensor([10., 10.,  9.,  9., -3.,  0., -1.,  1.])

    >>> index
    tensor([ 4688,  4497,  3411,   591,  5296,  7624,  3832, 12852])

    >>> e = {v: k for k, v in kb.entities.items()}

    >>> e[4688]
    'Liberia'

    >>> for i in index:
    ...     print(e[i.item()])
    Liberia
    Niger
    Sudan
    Nigeria
    Alien
    The Portrait of a Lady
    Girl with the Dragon Tattoo
    crossover

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
    ):
        self.k = k
        self.n = n
        self.max_tokens = max_tokens
        self.device = device
        self.kb_entities, _ = distillation_index(
            tokenizer=tokenizer, entities=entities, subwords_limit=subwords_limit
        )

        self.tokens, self.order = mapping_entities(
            tokenizer=tokenizer, max_tokens=self.max_tokens, entities=entities
        )

        if entities_to_distill is not None:
            self.filter_entities = {
                e.item(): True for e in self.kb_entities if e in entities_to_distill
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
        n_distributions = 0

        logits = collections.defaultdict(list)

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

                    for (entity, l, index) in self._top_k(model=model, **sample):

                        if entity in self.filter_entities:

                            logits[entity].append((l, index))
                            n_distributions += 1

                    bar.update(1)

                n_logits = len(logits)

                bar.set_description(
                    f"Updating Bert logits, {n_distributions} distributions, {n_logits} entities."
                )

                if n_distributions >= self.n:
                    break

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

        top_k_logits, top_k = bert_top_k(
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
