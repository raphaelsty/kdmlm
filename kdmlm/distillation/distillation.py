__all__ = ["Distillation"]

import collections
import random

import torch
from mkb import losses as mkb_losses

from ..utils import distillation_index, get_tensor_distillation
from .bert_logits import BertLogits
from .kb_logits import KbLogits


class Distillation:
    """Perform distillation between Bert and translationnal models.

    Examples
    --------

    >>> from kdmlm import distillation
    >>> from torch.utils import data

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> from kdmlm import datasets
    >>> from kdmlm import distillation

    >>> from mkb import datasets as mkb_datasets
    >>> from mkb import models as mkb_models

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> device = 'cpu'

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> bert_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> dataset = data.DataLoader(
    ...    dataset = datasets.KDDataset(
    ...        dataset = datasets.Sample(),
    ...        tokenizer = tokenizer,
    ...        sep = '|',
    ...     ),
    ...     collate_fn = datasets.Collator(tokenizer=tokenizer),
    ...     batch_size = 6,
    ... )

    >>> kb = mkb_datasets.Fb15k237(2, pre_compute=False, num_workers=0)

    >>> kb_model = mkb_models.TransE(
    ...     hidden_dim = 10,
    ...     gamma = 6,
    ...     entities = kb.entities,
    ...     relations = kb.relations,
    ... )

    >>> distillation = distillation.Distillation(
    ...     bert_model = bert_model,
    ...     kb_model = kb_model,
    ...     kb = kb,
    ...     dataset = dataset,
    ...     tokenizer = tokenizer,
    ...     entities = kb.entities,
    ...     k = 3,
    ...     n = 10,
    ...     max_tokens = 1,
    ...     subwords_limit = 1000,
    ...     device = device,
    ... )

    >>> sample = torch.tensor([
    ...    [11839, 0, 10],
    ...    [11190, 1, 11],
    ...    [11838, 2, 12]
    ... ])


    >>> distillation.distill_bert(
    ...     kb_model = kb_model,
    ...     sample = sample,
    ... )
    tensor(0.2421, grad_fn=<AddBackward0>)

    >>> entities = {value: key for key, value in kb.entities.items()}

    >>> entities[4186]
    'Bend It Like Beckham'

    >>> entities[5255]
    'Body of Lies'

    >>> entities[8772]
    '39th Canadian Parliament'

    >>> sentences = [
    ...     ("| Bend It Like Beckham | is the fist entity.", 4186),
    ...     ("| domestic partnership | is the second entity.", 5255),
    ...     ("| Gene Stupnitsky | is third entity.", 8772)
    ... ]

    >>> dataset = data.DataLoader(
    ...    dataset = datasets.KDDataset(
    ...        dataset = sentences,
    ...        tokenizer = tokenizer,
    ...        sep = '|',
    ...     ),
    ...     collate_fn = datasets.Collator(tokenizer=tokenizer),
    ...     batch_size = 3,
    ... )

    >>> for sample in dataset:
    ...     break

    >>> output = bert_model(
    ...     input_ids = sample["input_ids"],
    ...     attention_mask = sample["attention_mask"],
    ... )

    >>> distillation.distill_transe(
    ...     entities = sample["entity_ids"],
    ...     logits = output.logits,
    ...     labels = sample["labels"],
    ... )
    tensor(0.2229, grad_fn=<AddBackward0>)

    >>> candidates  = torch.tensor([11953,  8827,  7336,  2674,  4347,  9020])

    >>> distillation.distillation_sample(
    ...     candidates = candidates,
    ...     head = 574,
    ...     relation = 0,
    ...     tail = 1,
    ...     mode = 'head-batch',
    ...     heads = distillation.heads,
    ...     tails = distillation.tails,
    ... )
    tensor([[11953,     0,     1],
            [ 8827,     0,     1],
            [ 7336,     0,     1],
            [ 2674,     0,     1],
            [ 4347,     0,     1],
            [ 9020,     0,     1]])

    >>> distillation.distillation_sample(
    ...     candidates = candidates,
    ...     head = 574,
    ...     relation = 0,
    ...     tail = 1,
    ...     mode = 'tail-batch',
    ...     heads = distillation.heads,
    ...     tails = distillation.tails,
    ... )
    tensor([[  574,     0, 11953],
            [  574,     0,  8827],
            [  574,     0,  7336],
            [  574,     0,  2674],
            [  574,     0,  4347],
            [  574,     0,  9020]])

    >>> distillation = distillation.update_kb(
    ...     kb = kb,
    ...     kb_model = kb_model,
    ... )

    >>> distillation = distillation.update_bert(
    ...     model = bert_model,
    ...     tokenizer = tokenizer,
    ...     dataset = dataset,
    ... )

    """

    def __init__(
        self,
        bert_model,
        kb_model,
        kb,
        dataset,
        tokenizer,
        entities,
        k,
        n,
        do_distill_bert=True,
        do_distill_kg=True,
        max_tokens=15,
        subwords_limit=15,
        device="cuda",
        seed=42,
    ):
        self.do_distill_bert = do_distill_bert
        self.do_distill_kg = do_distill_kg
        self.device = device

        if self.do_distill_bert:

            self.bert_logits = BertLogits(
                model=bert_model,
                dataset=dataset,
                tokenizer=tokenizer,
                entities=entities,
                k=k,
                n=n,
                max_tokens=max_tokens,
                subwords_limit=subwords_limit,
                device=self.device,
            )

        if self.do_distill_kg:

            self.kb_logits = KbLogits(
                model=kb_model,
                dataset=kb,
                tokenizer=tokenizer,
                entities=kb.entities,
                k=k,
                n=n,
                device=self.device,
                subwords_limit=subwords_limit,
            )

        self.heads, self.tails = get_tensor_distillation([_ for _ in range(k * 2)])
        self.bert_entities, self.kb_entities = distillation_index(
            tokenizer=tokenizer,
            entities=entities,
            subwords_limit=subwords_limit,
        )
        self.bert_entities = self.bert_entities.to(self.device)

        random.seed(seed)

    @staticmethod
    def distillation_sample(candidates, head, relation, tail, mode, heads, tails):
        if mode == "head-batch":
            heads[:, 0] = candidates
            heads[:, 1] = relation
            heads[:, 2] = tail
            return heads.clone().detach()
        else:
            tails[:, 0] = head
            tails[:, 1] = relation
            tails[:, 2] = candidates
            return tails.clone().detach()

    def distill_bert(self, kb_model, sample):
        """Distill Bert to TransE from knowledge base."""
        samples, teacher_score = [], []

        if self.do_distill_bert:

            for h, r, t in sample:

                h, r, t = h.item(), r.item(), t.item()

                if h in self.bert_logits.logits:

                    l, c = random.choice(self.bert_logits.logits[h])
                    teacher_score.append(l)
                    samples.append(
                        self.distillation_sample(
                            candidates=c,
                            head=h,
                            relation=r,
                            tail=t,
                            mode="tail-batch",  # WRONG BUT JUST TO TRY
                            heads=self.heads,
                            tails=self.tails,
                        )
                    )

                if t in self.bert_logits.logits:

                    l, c = random.choice(self.bert_logits.logits[t])
                    teacher_score.append(l)
                    samples.append(
                        self.distillation_sample(
                            candidates=c,
                            head=h,
                            relation=r,
                            tail=t,
                            mode="head-batch",  # WRONG BUT JUST TO TRY
                            heads=self.heads,
                            tails=self.tails,
                        )
                    )

        loss = 0
        if teacher_score:
            teacher_score = torch.stack(teacher_score, dim=0)
            student_score = kb_model(torch.stack(samples, dim=0).to(self.device))
            loss += self.kl_divergence(teacher_score=teacher_score, student_score=student_score)

        return loss

    def distill_transe(self, entities, logits, labels):
        """Distill Transe to Bert."""
        student_score, teacher_score = [], []

        if self.do_distill_kg:

            mask_labels = labels != -100
            logits = logits[mask_labels]
            logits = torch.index_select(logits, 1, self.bert_entities)

            for p_e_c, e in zip(logits, entities):

                e = e.item()

                if e in self.kb_logits.logits:
                    top_k, candidates = random.choice(self.kb_logits.logits[e])
                else:
                    continue

                student_score.append(torch.index_select(input=p_e_c, dim=0, index=candidates))
                teacher_score.append(top_k)

        loss = 0
        if teacher_score:
            student_score = torch.stack(student_score, dim=0)
            teacher_score = torch.stack(teacher_score, dim=0)
            loss += self.kl_divergence(teacher_score=teacher_score, student_score=student_score)

        return loss

    def kl_divergence(self, teacher_score, student_score):
        error = 0
        if len(teacher_score) > 0:
            error = mkb_losses.KlDivergence()(
                student_score=student_score, teacher_score=teacher_score
            )
        return error

    def update_kb(self, kb, kb_model):
        """Updates distributions."""
        if self.do_distill_kg:
            self.kb_logits.logits = self.kb_logits.update(dataset=kb, model=kb_model)
        return self

    def update_bert(self, dataset, tokenizer, model):
        """Updates distributions."""
        if self.do_distill_bert:
            self.bert_logits.logits = self.bert_logits.update(dataset=dataset, model=model)
        return self
