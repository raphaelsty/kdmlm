import os
import random
from lib2to3.pgen2.tokenize import tokenize

import pandas as pd
import torch
import tqdm
from creme import stats
from kdmlm import distillation
from mkb import losses as mkb_losses
from mkb import sampling as mkb_sampling
from torch.utils import data
from transformers import Trainer

from ..datasets import WikiFb15k237Test
from ..distillation import Distillation
from ..utils import sentence_perplexity

__all__ = ["MlmTrainer"]


class EndTrainingException(Exception):
    """Stop training."""

    print("Done training.")
    pass


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
    ...     hidden_dim = 1,
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
    ...     sep='|',
    ...     mlm_probability = 0,
    ...     n_masks = 1,
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
    ...    model = model,
    ...    train_dataset = train_dataset,
    ...    tokenizer = tokenizer,
    ...    kb = kb,
    ...    kb_model = kb_model,
    ...    kb_evaluation = validation,
    ...    eval_every = 10,
    ...    negative_sampling_size = 10,
    ...    fit_kb_n_times = 1,
    ...    n = 10000,
    ...    top_k_size = 100,
    ...    update_top_k_every = 20,
    ...    alpha = 0.3,
    ...    seed = 42,
    ...    fit_bert = True,
    ...    fit_kb = True,
    ...    do_distill_kg = True,
    ...    do_distill_bert = True,
    ...    path_evaluation = 'evaluation.csv',
    ...    norm_loss = False,
    ...    max_step_bert = 10,
    ...    entities_to_distill = [1, 2, 3],
    ...    max_step_evaluation = 10,
    ... )

    >>> import kdmlm

    >>> try:
    ...     mlm_trainer.train()
    ... except kdmlm.mlm.mlm_trainer.EndTrainingException:
    ...     pass

    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        tokenizer,
        kb,
        kb_model,
        negative_sampling_size,
        kb_evaluation=None,
        eval_every=None,
        alpha=0.3,
        n=1000,
        top_k_size=100,
        update_top_k_every=1000,
        fit_kb_n_times=1,
        max_tokens=1,
        subwords_limit=1000,  # Maximum number of sub words to consider an entity.
        fit_bert=True,
        fit_kb=True,
        do_distill_bert=True,
        do_distill_kg=True,
        seed=42,
        path_evaluation=None,
        path_model_kb=None,
        lr_kb=0.00005,
        norm_loss=False,
        ewm_alpha=0.9997,
        max_step_bert=10 ** 10,
        temperature=1,
        entities_to_distill=None,
        max_step_evaluation=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=train_dataset.collate_fn,
            train_dataset=train_dataset,
        )

        self.subwords_limit = subwords_limit
        self.norm_loss = norm_loss
        self.max_step_bert = max_step_bert
        self.call = 0

        if self.norm_loss:

            self.ewm = {
                "link_prediction": stats.EWMean(alpha=ewm_alpha),
                "mlm": stats.EWMean(alpha=ewm_alpha),
                "distill_link_prediction": stats.EWMean(alpha=ewm_alpha),
                "distill_mlm": stats.EWMean(alpha=ewm_alpha),
            }

        self.alpha = alpha

        self.kb_model = kb_model.to(self.args.device)

        self.fit_bert = fit_bert
        self.fit_kb = fit_kb
        self.fit_kb_n_times = fit_kb_n_times

        self.path_model_kb = path_model_kb
        self.path_evaluation = path_evaluation
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
            device=self.args.device,
            temperature=temperature,
            entities_to_distill=entities_to_distill,
        )

        # Store kb evaluation scores
        self.top_k_size = top_k_size
        self.scores = []
        self.lr_kb = lr_kb
        self.kb_evaluation = kb_evaluation
        self.eval_every = eval_every

        self.tokenizer = tokenizer
        self.test_dataset = WikiFb15k237Test()

        self.max_step_evaluation = (
            max_step_evaluation if max_step_evaluation is not None else len(self.test_dataset)
        )

        self.metric_bert = stats.RollingMean(window_size=self.eval_every)
        self.metric_kb = stats.RollingMean(window_size=self.eval_every)
        self.metric_kb_kl = stats.RollingMean(window_size=self.eval_every)
        self.metric_bert_kl = stats.RollingMean(window_size=self.eval_every)
        # Perplexity reset every time we call evaluation of Bert:
        self.metric_perplexity = stats.RollingMean(window_size=self.max_step_evaluation)

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
                        )

                    self.step_bert += 1

                    distillation_loss = self.distillation.distill_bert(
                        kb_model=self.kb_model,
                        sample=sample,
                    )

                    if distillation_loss != 0:

                        self.metric_kb_kl.update(distillation_loss.item())

                if self.distillation.do_distill_bert and self.fit_kb:

                    if self.norm_loss:

                        loss = self.normalize_loss(
                            loss=loss, distillation_loss=distillation_loss, bert=False
                        )

                    else:

                        loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

                elif self.distillation.do_distill_bert and not self.fit_kb:

                    loss = distillation_loss

                if loss != 0:

                    loss.backward()
                    self.kb_optimizer.step()
                    self.kb_optimizer.zero_grad()

                if (self.step_kb + 1) % self.update_top_k_every == 0:
                    self.distillation.update_kb(kb=self.kb, kb_model=self.kb_model)

        return self

    def training_step(self, model, inputs):
        """Training step."""
        self.call += 1
        if self.call > self.max_step_bert:
            raise EndTrainingException

        loss = 0
        distillation_loss = 0

        self.training_step_kb(model=model)

        if self.fit_bert or self.distillation.do_distill_kg:

            # Eval perplexity every 100 steps:

            if self.eval_every is not None:

                if (self.call + 1) % self.eval_every == 0:

                    self.evaluation(model=model)

            model = model.train()
            inputs.pop("mask")

            # classic MLM
            if "entity_ids" in inputs:
                entity_ids = inputs.pop("entity_ids")
                mlm_mode = False
            else:
                mlm_mode = True

            labels = inputs["labels"]
            inputs = self._prepare_inputs(inputs)

            if self.fit_bert:

                loss, outputs = self.compute_loss(model, inputs=inputs, return_outputs=True)

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                self.metric_bert.update(loss.item())

            if not self.fit_bert and self.distillation.do_distill_kg and not mlm_mode:

                outputs = model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )

            if self.distillation.do_distill_kg and not mlm_mode:

                distillation_loss = self.distillation.distill_transe(
                    entities=entity_ids,
                    logits=outputs.logits,
                    labels=labels,
                )

                if distillation_loss != 0:

                    self.metric_bert_kl.update(distillation_loss.item())

            if self.distillation.do_distill_kg and self.fit_bert and not mlm_mode:

                if self.norm_loss:

                    loss = self.normalize_loss(
                        loss=loss, distillation_loss=distillation_loss, bert=True
                    )

                else:

                    loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

            elif self.distillation.do_distill_kg and not self.fit_bert:

                loss = distillation_loss

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

    def evaluation(self, model):
        """Eval KB model."""

        scores_valid = {}
        scores_test = {}

        if self.fit_kb or self.distillation.do_distill_bert:

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

            if self.path_model_kb is not None:

                self.kb_model.cpu().save(
                    os.path.join(self.path_model_kb, f"{self.kb_model.name}_{self.step_kb}")
                )

                self.kb_model.to(self.args.device)

        if self.fit_bert or self.distillation.do_distill_kg:

            bar = tqdm.tqdm(
                enumerate(self.test_dataset),
                position=0,
                desc="Evaluating PPL",
                total=self.max_step_evaluation,
            )

            for step_evaluation_bert, sentence in bar:

                if step_evaluation_bert > self.max_step_evaluation:
                    break

                self.metric_perplexity.update(
                    sentence_perplexity(
                        model=model,
                        tokenizer=self.tokenizer,
                        sentence=sentence,
                        device=self.args.device,
                    )
                )

                bar.set_description(f"Evaluating PPL {self.metric_perplexity.get():2f}")

        if self.path_evaluation is not None:
            self.export_to_csv(name="valid", score=scores_valid, step=self.step_kb)
            self.export_to_csv(name="test", score=scores_test, step=self.step_kb)

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
        score["perplexity"] = self.metric_perplexity.get()
        self.scores.append(pd.DataFrame.from_dict(score, orient="index").T)
        pd.concat(self.scores, axis="rows").to_csv(self.path_evaluation, index=False)

    def normalize_loss(self, loss, distillation_loss, bert):
        """Normalize losses."""

        if distillation_loss == 0:
            return loss

        if bert:
            self.ewm["mlm"].update(loss.detach().item())
            self.ewm["distill_mlm"].update(distillation_loss.detach().item())
        else:
            self.ewm["link_prediction"].update(loss.detach().item())
            self.ewm["distill_link_prediction"].update(distillation_loss.detach().item())

        weight = 1 / (1 + self.alpha)

        try:
            if bert:
                scale = self.ewm["mlm"].get() / self.ewm["distill_mlm"].get()
            else:
                scale = (
                    self.ewm["link_prediction"].get() / self.ewm["distill_link_prediction"].get()
                )
        except ZeroDivisionError:
            return loss

        return weight * (loss + self.alpha * scale * distillation_loss)
