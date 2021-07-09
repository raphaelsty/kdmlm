__all__ = ["Distillation"]

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from mkb import losses as mkb_losses

from ..utils import distillation_index, get_tensor_distillation
from .bert_logits import BertLogits
from .kb_logits import KbLogits


class Distillation:
    """Perform distillation between Bert and translationnal models.

    Examples
    --------

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
        temperature=1,
        do_distill_bert=True,
        do_distill_kg=True,
        max_tokens=15,
        subwords_limit=15,
        device="cuda",
        seed=42,
        entities_to_distill=None,
    ):
        self.do_distill_bert = do_distill_bert
        self.do_distill_kg = do_distill_kg
        self.device = device
        self.temperature = temperature

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
                entities_to_distill=entities_to_distill,
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
                entities_to_distill=entities_to_distill,
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
        """Distill Bert to TransE from knowledge base.

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
        ...     temperature = 10,
        ... )

        >>> entities_bert = list(distillation.bert_logits.logits.keys())

        >>> entities_bert
        [11839, 11190, 11838, 6133, 12603, 5136, 3805, 4721, 1197, 2421, 12127]

        >>> entities = {v: k for k, v in kb.entities.items()}

        >>> for e in entities_bert:
        ...     print(entities[e])
        Zachary Taylor
        James Buchanan
        Confederate States of America
        William Tecumseh Sherman
        Mississippi River
        Ulysses S. Grant
        James Madison
        West Virginia
        Nevada
        Illinois
        nationalism

        >>> for _, candidates in distillation.bert_logits.logits[2421]:
        ...     for c in candidates.tolist():
        ...         print(entities[c])
        Ohio University
        Ohio State Buckeyes football
        Ohio
        Creighton University
        Joshua Redman
        Helen

        >>> sample = torch.tensor([
        ...    [11839, 0, 2421],
        ...    [11839, 1, 2421]
        ... ])


        >>> distillation.distill_bert(kb_model = kb_model, sample = sample)
        """
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
                            mode="head-batch",
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
                            mode="tail-batch",
                            heads=self.heads,
                            tails=self.tails,
                        )
                    )

        loss = 0
        if teacher_score:
            teacher_score = torch.stack(teacher_score, dim=0)
            student_score = kb_model(torch.stack(samples, dim=0).to(self.device))
            loss += self.kl_divergence(
                teacher_score=teacher_score, student_score=student_score, T=self.temperature
            )

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

            loss += self.kl_divergence(
                teacher_score=teacher_score, student_score=student_score, T=self.temperature
            )

        return loss

    def update_kb(self, kb, kb_model):
        """Updates distributions."""
        if self.do_distill_kg:
            self.kb_logits.logits = self.kb_logits.update(dataset=kb, model=kb_model)
        return self

    def update_bert(self, dataset, model):
        """Updates distributions."""
        if self.do_distill_bert:
            self.bert_logits.logits = self.bert_logits.update(dataset=dataset, model=model)
        return self

    @staticmethod
    def kl_divergence(teacher_score, student_score, T):
        error = 0
        if len(teacher_score) > 0:
            error = mkb_losses.KlDivergence()(
                student_score=student_score,
                teacher_score=teacher_score,
                T=T,
            )
        return error
