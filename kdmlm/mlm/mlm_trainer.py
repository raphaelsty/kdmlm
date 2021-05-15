import os
import random

import pandas as pd
import torch
from creme import stats
from mkb import losses as mkb_losses
from mkb import sampling as mkb_sampling
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import Trainer

from ..distillation import Distillation

__all__ = ["MlmTrainer"]


class MlmTrainer(Trainer):
    """Custom trainer to distill knowledge to bert from knowledge graphs embeddings.

    Parameters
    ----------

        knowledge: mkb model.
        alpha (float): Weight to give to knowledge distillation.


    Examples
    --------

    >>> from kdmlm import mlm
    >>> from kdmlm import datasets

    >>> from mkb import datasets as mkb_datasets
    >>> from mkb import models as mkb_models
    >>> from mkb import evaluation

    >>> from transformers import BertTokenizer
    >>> from transformers import BertForMaskedLM

    >>> from transformers import DataCollatorForLanguageModeling
    >>> from transformers import LineByLineTextDataset
    >>> from transformers import TrainingArguments

    >>> import torch
    >>> _ = torch.manual_seed(42)

    >>> device = 'cpu'

    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    >>> kb = mkb_datasets.Fb15k237(10, pre_compute = False, num_workers=0)
    >>> kb.test = kb.test[:2]
    >>> kb.test = kb.test[:1]

    >>> kb_model = mkb_models.TransE(
    ...     entities = kb.entities,
    ...     relations = kb.relations,
    ...     hidden_dim = 2,
    ...     gamma = 8
    ... )

    >>> validation = evaluation.Evaluation(
    ...     true_triples = kb.train + kb.valid + kb.test,
    ...     entities = kb.entities,
    ...     relations = kb.relations,
    ...     batch_size = 8,
    ...     device = device,
    ... )

    >>> train_dataset = datasets.KDDataset(
    ...     dataset=datasets.Sample(),
    ...     tokenizer=tokenizer,
    ...     sep='|'
    ... )

    >>> training_args = TrainingArguments(
    ...     output_dir = f'./checkpoints',
    ...     overwrite_output_dir = True,
    ...     num_train_epochs = 2,
    ...     per_device_train_batch_size = 10,
    ...     save_steps = 500,
    ...     save_total_limit = 1,
    ...     do_train = True,
    ...     do_predict = True,
    ... )

    >>> mlm_trainer = MlmTrainer(
    ...    args = training_args,
    ...    data_collator = train_dataset.collate_fn,
    ...    model = model,
    ...    train_dataset = train_dataset,
    ...    tokenizer = tokenizer,
    ...    kb = kb,
    ...    kb_model = kb_model,
    ...    kb_evaluation = validation,
    ...    eval_kb_every = 30,
    ...    negative_sampling_size = 10,
    ...    fit_kb_n_times = 2,
    ...    n = 10,
    ...    top_k_size = 100,
    ...    update_top_k_every = 20,
    ...    alpha = 0.3,
    ...    seed = 42,
    ...    fit_bert = True,
    ...    fit_kb = True,
    ...    do_distill_kg = True,
    ...    do_distill_bert = True,
    ...    wiki_mode=True,
    ...    path_score_kb = 'evaluation.csv',
    ... )

    >>> mlm_trainer.train()

    """

    def __init__(
        self,
        model,
        args,
        data_collator,
        train_dataset,
        tokenizer,
        kb,
        kb_model,
        negative_sampling_size,
        kb_evaluation=None,
        eval_kb_every=None,
        alpha=0.3,
        n=1000,
        top_k_size=100,
        update_top_k_every=1000,
        fit_kb_n_times=1,
        max_tokens=15,
        subwords_limit=15,  # Maximum number of sub words to consider an entity.
        fit_bert=True,
        fit_kb=True,
        do_distill_bert=True,
        do_distill_kg=True,
        seed=42,
        path_score_kb=None,
        path_model_kb=None,
        lr_kb=0.00005,
        wiki_mode=True,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=train_dataset.collate_fn,
            train_dataset=train_dataset,
        )

        self.wiki_mode = wiki_mode
        self.subwords_limit = subwords_limit

        if self.wiki_mode:

            self.distill_wiki = DataLoader(
                train_dataset,
                collate_fn=train_dataset.collate_fn,
                batch_size=kb.batch_size,
            )

        self.alpha = alpha

        self.kb_model = kb_model.to(self.args.device)

        self.fit_bert = fit_bert
        self.fit_kb = fit_kb
        self.fit_kb_n_times = fit_kb_n_times

        self.path_model_kb = path_model_kb
        self.path_score_kb = path_score_kb
        self.results = []

        if self.path_model_kb is not None:
            if not os.path.exists(self.path_model_kb):
                os.makedirs(self.path_model_kb)

        random.seed(seed)

        # Link prediction task
        self.negative_sampling = mkb_sampling.NegativeSampling(
            size=negative_sampling_size,
            train_triples=kb.train,
            entities=kb.entities,
            relations=kb.relations,
            seed=42,
        )

        self.kb_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kb_model.parameters()),
            lr=lr_kb,
        )

        self.kb = kb
        self.update_top_k_every = update_top_k_every

        self.step_kb = 0
        self.step_bert = 0

        self.dataset_logits = data.DataLoader(
            dataset=train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_size=100,
        )

        self.distillation = Distillation(
            bert_model=model,
            kb_model=kb_model,
            kb=kb,
            dataset=self.dataset_logits,
            tokenizer=tokenizer,
            entities=kb.entities,
            k=top_k_size,
            n=n,
            do_distill_bert=do_distill_bert,
            do_distill_kg=do_distill_kg,
            max_tokens=max_tokens,
            subwords_limit=subwords_limit,
            wiki_mode=wiki_mode,
            device=self.args.device,
        )

        # Store kb evaluation scores
        self.top_k_size = top_k_size
        self.scores = []
        self.lr_kb = lr_kb
        self.kb_evaluation = kb_evaluation
        self.eval_kb_every = eval_kb_every

        self.metric_bert = stats.RollingMean(window_size=self.eval_kb_every)
        self.metric_kb = stats.RollingMean(window_size=self.eval_kb_every)
        self.metric_kb_kl = stats.RollingMean(window_size=self.eval_kb_every)
        self.metric_bert_kl = stats.RollingMean(window_size=self.eval_kb_every)

    @staticmethod
    def print_scores(step, name, scores):
        print("\n")
        print(f"{name} - {step}")
        for key, value in scores.items():
            print(f"\t{key}: {value:3f}")
        print("\n")

    def training_step_kb(self, model):
        """Update kb model."""
        if self.fit_kb or self.distillation.do_distill_bert:

            for _ in range(self.fit_kb_n_times):

                self.step_kb += 1

                loss = 0
                distillation_loss = 0

                data = next(self.kb)
                sample, weight, mode = data["sample"], data["weight"], data["mode"]

                if self.fit_kb:

                    loss += self.link_prediction(sample=sample, weight=weight, mode=mode)

                    if loss != 0:
                        self.metric_kb.update(loss.item())

                if self.distillation.do_distill_bert:

                    if (self.step_bert + 1) % self.update_top_k_every == 0:

                        self.distillation.update_bert(
                            model=model,
                            dataset=self.dataset_logits,
                            tokenizer=self.tokenizer,
                        )

                    self.step_bert += 1

                    if self.wiki_mode:

                        distillation_loss = 0

                        for sentences in self.distill_wiki:
                            break

                        for mode in ["head-batch", "tail-batch"]:

                            distillation_loss += self.distillation.distill_bert_wiki(
                                kb_model=self.kb_model,
                                entity_ids=sentences["entity_ids"],
                                mode=mode,
                            )

                    else:

                        distillation_loss = self.distillation.distill_bert(
                            kb_model=self.kb_model,
                            sample=sample,
                        )

                    if distillation_loss != 0:
                        self.metric_kb_kl.update(distillation_loss.item())

                loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

                if loss != 0:

                    loss.backward()
                    self.kb_optimizer.step()
                    self.kb_optimizer.zero_grad()

                if self.kb_evaluation is not None:
                    self.link_prediction_evaluation()

                if (self.step_kb + 1) % self.update_top_k_every == 0:
                    self.distillation.update_kb(kb=self.kb, kb_model=self.kb_model)

        return self

    def training_step(self, model, inputs):
        """Training step."""

        self.training_step_kb(model=model)
        loss = 0
        distillation_loss = 0

        if self.fit_bert or self.distillation.do_distill_kg:

            model.train()

            inputs.pop("mask")
            entity_ids = inputs.pop("entity_ids")
            labels = inputs["labels"]
            inputs = self._prepare_inputs(inputs)

            if self.fit_bert:

                loss, outputs = self.compute_loss(model, inputs=inputs, return_outputs=True)

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.metric_bert.update(loss.item())

            if not self.fit_bert and self.distillation.do_distill_kg:

                outputs = model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )

            if self.distillation.do_distill_kg:

                distillation_loss = self.distillation.distill_transe(
                    entities=entity_ids,
                    logits=outputs.logits,
                    labels=labels,
                )

                if distillation_loss != 0:
                    self.metric_bert_kl.update(distillation_loss.item())

                model = model.eval()

            loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

            if loss != 0:
                loss.backward()
                loss = loss.detach()

        return loss

    def link_prediction(self, sample, weight, mode):
        """"Method dedicated to link prediction task."""
        negative_sample = self.negative_sampling.generate(sample=sample, mode=mode)

        sample = sample.to(self.args.device)
        negative_sample = negative_sample.to(self.args.device)

        positive_score = self.kb_model(sample)
        negative_score = self.kb_model(
            sample=sample,
            negative_sample=negative_sample,
            mode=mode,
        )

        loss = mkb_losses.Adversarial(0.5)(
            positive_score,
            negative_score,
            weight=weight.to(self.args.device),
        )
        return loss

    def link_prediction_evaluation(self):
        """Eval KB model."""
        if (self.step_kb + 1) % self.eval_kb_every == 0:

            scores_valid = self.kb_evaluation.eval(
                model=self.kb_model,
                dataset=self.kb.valid,
            )

            scores_test = self.kb_evaluation.eval(
                model=self.kb_model,
                dataset=self.kb.test,
            )

            self.print_scores(step=self.step_kb, name="valid", scores=scores_valid)
            self.print_scores(step=self.step_kb, name="test", scores=scores_test)

            if self.path_score_kb is not None:
                self.export_to_csv(name="valid", score=scores_valid, step=self.step_kb)
                self.export_to_csv(name="test", score=scores_test, step=self.step_kb)

            if self.path_model_kb is not None:
                self.kb_model.cpu().save(
                    os.path.join(self.path_model_kb, f"{self.kb_model.name}_{self.step_kb}")
                )
                self.kb_model.to(self.args.device)

        return self

    def export_to_csv(self, name, score, step):
        """Export scores as a csv file."""
        score["step"] = step
        score["alpha"] = self.alpha
        score["name"] = name
        score["k"] = self.top_k_size
        score["lr_kb"] = self.lr_kb
        score["bert_kl"] = self.metric_bert_kl.get()
        score["kb_kl"] = self.metric_kb_kl.get()
        score["bert_loss"] = self.metric_bert.get()
        score["kb_loss"] = self.metric_kb.get()
        self.scores.append(pd.DataFrame.from_dict(score, orient="index").T)
        pd.concat(self.scores, axis="rows").to_csv(self.path_score_kb, index=False)
