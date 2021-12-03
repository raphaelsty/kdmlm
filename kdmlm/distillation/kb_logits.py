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
    >>> from kdmlm import datasets
    >>> import pickle

    >>> with open("./data/TransE_76299", "rb") as input_model:
    ...     model = pickle.load(input_model)

    >>> from transformers import DistilBertTokenizer

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> dataset = datasets.Fb15k237(30, pre_compute=False, num_workers=0, shuffle=False)

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    >>> kb_logits = distillation.KbLogits(
    ...     dataset = dataset,
    ...     model = model,
    ...     entities = dataset.entities,
    ...     tokenizer = tokenizer,
    ...     subwords_limit = 1,
    ...     k = 10,
    ...     n = 100,
    ...     device = 'cpu',
    ...     average = True
    ... )

    >>> for distribution in kb_logits.logits.values():
    ...     assert len(distribution) == 1

    >>> logits = {}
    >>> for e, distributions in kb_logits.logits.items():
    ...     logits[model.entities[e]] = ([model.entities[x.item()] for x in distributions[0][0]])

    >>> for e in logits["harpsichord"][:5]:
    ...     print(e)
    violin
    cello
    guitar
    piano
    viola

    """

    def __init__(
        self,
        dataset,
        model,
        entities,
        tokenizer,
        subwords_limit,
        k,
        n,
        device,
        entities_to_distill=None,
        average=False,
    ):
        self.k = k
        self.n = n
        self.device = device
        self.average = average
        self.n_entities = len(entities)

        kb_entities, _ = distillation_index(
            tokenizer=tokenizer, entities=entities, subwords_limit=subwords_limit
        )

        # Entities filtered with subwords_limit
        self.filter_entities = {e.item(): True for e in kb_entities}

        if entities_to_distill is not None:
            self.filter_entities = {
                e.item(): True for e in kb_entities if e.item() in entities_to_distill
            }
        else:
            self.filter_entities = {e.item(): True for e in kb_entities}

        self.heads, self.tails = get_tensor_distillation(kb_entities)
        self.heads, self.tails = self.heads.to(self.device), self.tails.to(self.device)

        self.logits = self.update(dataset=dataset, model=model)

    def update(self, dataset, model):

        logits = collections.defaultdict(list)

        if self.average:
            average_logits = {}
            frequency = collections.defaultdict(int)

        n_distributions = 0

        bar = tqdm.tqdm(
            range(min(self.n // dataset.batch_size, len(dataset.train) // dataset.batch_size)),
            desc=f"Updating kb logits, {n_distributions} distributions, {len(logits)} entities.",
            position=0,
        )

        with torch.no_grad():

            for _ in bar:

                sample = next(dataset)["sample"]

                for h, r, t in sample:

                    h, r, t = h.item(), r.item(), t.item()

                    if h in self.filter_entities:

                        if self.average:

                            score = self._scores(
                                model=model,
                                h=h,
                                r=r,
                                t=t,
                                tensor_distillation=self.heads,
                                mode="head-batch",
                            )

                            if h in logits:
                                average_logits[h] += score
                            else:
                                average_logits[h] = score

                            frequency[h] += 1

                        else:

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

                        if self.average:

                            score = self._scores(
                                model=model,
                                h=h,
                                r=r,
                                t=t,
                                tensor_distillation=self.tails,
                                mode="tail-batch",
                            )

                            if t in logits:
                                average_logits[t] += score
                            else:
                                average_logits[t] = score

                            frequency[t] += 1

                        else:

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

        if self.average:
            logits = self._logits_from_average(average_logits=average_logits, frequency=frequency)

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
        ...     device = 'cpu',
        ...     average = False
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

    def _scores(self, model, h, r, t, tensor_distillation, mode):
        """Scores for average top k."""
        tensor_distillation[:, 1] = r

        if mode == "head-batch":
            tensor_distillation[:, 2] = t

        elif mode == "tail-batch":
            tensor_distillation[:, 0] = h

        return model(tensor_distillation.to(self.device)).flatten()

    def _logits_from_average(self, average_logits, frequency):
        """Compute logits i.e index and scores from average logits."""
        logits = collections.defaultdict(list)

        kb_entities = self.heads[:, 0]

        for e, score in average_logits.items():

            score /= frequency[e]

            # Concatenate top k index and random k index
            index = torch.cat(
                [
                    torch.argsort(score, dim=0, descending=True)[: self.k],
                    torch.randint(low=0, high=len(score) - 1, size=(self.k,)).to(self.device),
                ]
            )

            score = torch.index_select(score, 0, index)

            # Retrieve entities from top k index.
            index = torch.index_select(
                kb_entities,
                0,
                index,
            ).to(self.device)

            logits[e].append((index, score))

        return logits
