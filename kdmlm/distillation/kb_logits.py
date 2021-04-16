__all__ = ["KbLogits"]

import collections

import torch
import tqdm

from ..utils import get_tensor_distillation


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

    >>> from mkb import models
    >>> from kdmlm import distillation
    >>> from mkb import datasets

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Fb15k237(30, pre_compute=False, num_workers=0, shuffle=False)

    >>> model = models.TransE(
    ...    entities = dataset.entities,
    ...    relations = dataset.relations,
    ...    hidden_dim = 10,
    ...    gamma = 5,
    ... )

    >>> kb_logits = distillation.KbLogits(
    ...     dataset = dataset,
    ...     model = model,
    ...     entities = dataset.entities,
    ...     k = 2,
    ...     n = 1000,
    ...     device = 'cpu'
    ... )

    >>> len(kb_logits.logits.keys())
    876

    """

    def __init__(self, dataset, model, entities, k=100, n=1000, device="cuda"):
        self.k = k
        self.n = n
        self.device = device
        self.n_entities = len(entities)
        self.heads, self.tails = get_tensor_distillation([_ for _ in range(self.n_entities)])
        self.logits = self.update(dataset=dataset, model=model)

    def update(self, dataset, model):
        logits = collections.defaultdict(list)

        bar = tqdm.tqdm(
            range(min(self.n // dataset.batch_size, len(dataset.train))),
            desc="Updating kb logits",
            position=0,
        )

        for _ in bar:

            sample = next(dataset)["sample"]

            with torch.no_grad():

                for h, t, heads_score, tails_score, heads_index, tails_index in self._top_k(
                    model, sample
                ):
                    logits[h].append((heads_score, heads_index))
                    logits[t].append((tails_score, tails_index))

        return logits

    def _top_k(self, model, sample):
        """Compute top k plus random k for heads and tails of an input sample of triplets.

        Parameters
        ----------
            model (mkb.Models): Translationnal model.
            sample (torch.tensor): Input sample.

        Examples
        --------

        >>> from mkb import models
        >>> from kdmlm import distillation
        >>> from mkb import datasets

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> dataset = datasets.Fb15k237(2, pre_compute = False, num_workers = 0, shuffle = False)

        >>> model = models.TransE(
        ...    entities = dataset.entities,
        ...    relations = dataset.relations,
        ...    hidden_dim = 10,
        ...    gamma = 5,
        ... )

        >>> kb_logits = distillation.KbLogits(
        ...     dataset = dataset,
        ...     model = model,
        ...     entities = dataset.entities,
        ...     k = 2,
        ...     n = 10,
        ...     device = 'cpu'
        ... )

        >>> sample = torch.tensor([
        ...    [ 2067,    17, 13044],
        ...    [ 4867,   140,  1984]])

        >>> for h, t, heads_score, tails_score, heads_index, tails_index in kb_logits._top_k(
        ...     model = model, sample = sample):
        ...     break

        >>> h
        2067

        >>> t
        13044

        >>> heads_score
        tensor([ 2.8958,  2.6915, -0.7853, -1.7397], grad_fn=<IndexSelectBackward>)

        >>> tails_score
        tensor([ 2.9989,  2.8445, -1.2583, -0.8696], grad_fn=<IndexSelectBackward>)

        >>> heads_index
        tensor([ 8722, 10923,  1302,  3299])

        >>> tails_index
        tensor([9239, 5352, 7835, 5895])

        >>> model(torch.tensor([[2067, 17, 9239]]))
        tensor([[2.9989]], grad_fn=<ViewBackward>)

        >>> model(torch.tensor([[8722, 17, 13044]]))
        tensor([[2.8958]], grad_fn=<ViewBackward>)

        """
        for h, r, t in sample:

            h, r, t = h.item(), r.item(), t.item()

            self.heads[:, 1] = r
            self.heads[:, 2] = t

            self.tails[:, 0] = h
            self.tails[:, 1] = r

            heads_score = model(self.heads.to(self.device)).flatten()
            tails_score = model(self.tails.to(self.device)).flatten()

            heads_index = torch.cat(
                [
                    torch.argsort(heads_score, dim=0, descending=True)[: self.k].detach().cpu(),
                    torch.randint(low=0, high=self.n_entities - 1, size=(self.k,)),
                ]
            )

            tails_index = torch.cat(
                [
                    torch.argsort(tails_score, dim=0, descending=True)[: self.k].detach().cpu(),
                    torch.randint(low=0, high=self.n_entities - 1, size=(self.k,)),
                ]
            )

            heads_score = torch.index_select(heads_score, 0, heads_index)
            tails_score = torch.index_select(tails_score, 0, tails_index)

            yield h, t, heads_score, tails_score, heads_index, tails_index
