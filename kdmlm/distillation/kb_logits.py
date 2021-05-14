__all__ = ["KbLogits"]

import collections

import torch
import tqdm

from ..utils import distillation_index, get_tensor_distillation


class KbLogits:
    """Compute Kb logits for a given Kb.

    Parameters
    ----------
        entities (dict): Entities of the knowledge base.
        n (int): Number of batch to consider for generating distributions.
        k (int): Size of the top k.
        device (str): Device, cuda or cpu.

    Examples
    --------

    >>> from kdmlm import distillation

    >>> from mkb import models
    >>> from mkb import datasets

    >>> from transformers import DistilBertTokenizer

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Fb15k237(30, pre_compute=False, num_workers=0, shuffle=False)

    >>> model = models.TransE(
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    hidden_dim = 10,
    ...    gamma = 5,
    ... )

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> kb_logits = distillation.KbLogits(
    ...     dataset = dataset,
    ...     model = model,
    ...     entities = dataset.entities,
    ...     tokenizer = tokenizer,
    ...     subwords_limit = 1,
    ...     k = 1,
    ...     n = 1000,
    ...     device = 'cpu'
    ... )

    >>> len(kb_logits.logits.keys())
    122

    """

    def __init__(self, dataset, model, entities, tokenizer, subwords_limit, k, n, device):
        self.k = k
        self.n = n
        self.device = device
        self.n_entities = len(entities)

        kb_entities, _ = distillation_index(
            tokenizer=tokenizer, entities=entities, subwords_limit=subwords_limit
        )

        # Entities filtered with subwords_limit
        self.filter_entities = {e.item(): True for e in kb_entities}

        self.heads, self.tails = get_tensor_distillation(kb_entities)

        self.logits = self.update(dataset=dataset, model=model)

    def update(self, dataset, model):
        logits = collections.defaultdict(list)

        n_distributions = 0

        bar = tqdm.tqdm(
            range(min(self.n // dataset.batch_size, len(dataset.train))),
            desc=f"Updating kb logits, {n_distributions} distributions, {len(logits)} entities.",
            position=0,
        )

        with torch.no_grad():

            for _ in bar:

                sample = next(dataset)["sample"]

                for h, r, t in sample:

                    h, r, t = h.item(), r.item(), t.item()

                    if h in self.filter_entities:

                        score, index = self._top_k(
                            model=model,
                            h=h,
                            r=r,
                            t=t,
                            tensor_distillation=self.heads,
                            mode="head-batch",
                        )

                        logits[h].append((score, index))

                        n_distributions += 1

                    if t in self.filter_entities:

                        score, index = self._top_k(
                            model=model,
                            h=h,
                            r=r,
                            t=t,
                            tensor_distillation=self.tails,
                            mode="tail-batch",
                        )

                        logits[t].append((score, index))

                        n_distributions += 1

                bar.set_description(
                    f"Updating kb logits, {n_distributions} distributions, {len(logits)} entities."
                )

        return logits

    def _top_k(self, model, h, r, t, tensor_distillation, mode):
        """Compute top k plus random k for heads and tails of an input sample of triplets.

        Parameters
        ----------
            model (mkb.Models): Translationnal model.
            sample (torch.tensor): Input sample.

        Examples
        --------

        >>> from kdmlm import distillation

        >>> from mkb import models
        >>> from mkb import datasets

        >>> from transformers import DistilBertTokenizer

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Fb15k237(2, pre_compute = False, num_workers = 0, shuffle = False)

        >>> model = models.TransE(
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    hidden_dim = 10,
        ...    gamma = 5,
        ... )

        >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        >>> kb_logits = distillation.KbLogits(
        ...     dataset = dataset,
        ...     model = model,
        ...     entities = dataset.entities,
        ...     tokenizer = tokenizer,
        ...     subwords_limit = 2,
        ...     k = 2,
        ...     n = 10,
        ...     device = 'cpu'
        ... )

        >>> sample = torch.tensor([
        ...    [ 2067,    17, 13044],
        ...    [ 4867,   140,  1984]])

        >>> logits, index  = kb_logits._top_k(model = model, h = 2067, r = 17, t = 13044,
        ...     tensor_distillation = kb_logits.heads, mode = "head-batch")

        >>> logits
        tensor([ 2.8958,  2.3543, -2.8897, -0.6057], grad_fn=<IndexSelectBackward>)

        >>> index
        tensor([ 8722,   976, 11999, 10242])

        >>> logits, index = kb_logits._top_k(model = model, h = 2067, r = 17, t = 13044,
        ...     tensor_distillation = kb_logits.tails, mode = "tail-batch")

        >>> logits
        tensor([ 2.7046,  2.5926, -1.4249,  0.2401], grad_fn=<IndexSelectBackward>)

        >>> index
        tensor([ 2655, 13315,  6234,  3326])

        >>> model(torch.tensor([[8722, 17, 13044]]))
        tensor([[2.8958]], grad_fn=<ViewBackward>)

        >>> model(torch.tensor([[2067, 17, 2655]]))
        tensor([[2.7046]], grad_fn=<ViewBackward>)

        """
        tensor_distillation[:, 1] = r

        if mode == "head-batch":
            tensor_distillation[:, 2] = t

        elif mode == "tail-batch":
            tensor_distillation[:, 0] = h

        score = model(tensor_distillation.to(self.device)).flatten()

        # Concatenate top k index and random k index
        index = torch.cat(
            [
                torch.argsort(score, dim=0, descending=True)[: self.k],
                torch.randint(low=0, high=len(tensor_distillation) - 1, size=(self.k,)).to(
                    self.device
                ),
            ]
        )

        score = torch.index_select(score, 0, index)

        # Retrieve entities from top k index.
        index = torch.index_select(
            tensor_distillation[:, 0] if mode == "head-batch" else tensor_distillation[:, 2],
            0,
            index,
        ).to(self.device)

        return score, index
