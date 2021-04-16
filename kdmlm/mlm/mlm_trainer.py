import random

import torch
from mkb import losses as mkb_losses
from mkb import sampling as mkb_sampling
from torch.utils import data
from transformers import Trainer

from ..distillation import Distillation

__all__ = ["MlmTrainer"]


class MlmTrainer(Trainer):
    """Custom trainer to distill knowledge to bert from knowledge graphs embeddings.

    Parameters:
    -----------

        knowledge: mkb model.
        alpha (float): Weight to give to knowledge distillation.


    Example:

        >>> from kdmlm import mlm
        >>> from kdmlm import datasets

        >>> from mkb import datasets as mkb_datasets
        >>> from mkb import models as mkb_models

        >>> from transformers import BertTokenizer
        >>> from transformers import BertForMaskedLM

        >>> from transformers import DataCollatorForLanguageModeling
        >>> from transformers import LineByLineTextDataset
        >>> from transformers import TrainingArguments

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        >>> kb = mkb_datasets.Fb15k237(10, pre_compute = False, num_workers=0)

        >>> kb_model = mkb_models.TransE(
        ...     entities = kb.entities,
        ...     relations = kb.relations,
        ...     hidden_dim = 2,
        ...     gamma = 8
        ... )

        >>> train_dataset = datasets.KDDataset(
        ...     dataset=datasets.Sample(),
        ...     tokenizer=tokenizer,
        ...     sep='|'
        ... )

        >>> data_collator = datasets.Collator(tokenizer=tokenizer)

        >>> training_args = TrainingArguments(output_dir = f'./checkpoints',
        ... overwrite_output_dir = True, num_train_epochs = 3,
        ... per_device_train_batch_size = 10, save_steps = 500, save_total_limit = 1,
        ... do_train = True, do_predict = True)

        >>> mlm_trainer = MlmTrainer(
        ...    args = training_args,
        ...    data_collator = data_collator,
        ...    model = model,
        ...    train_dataset = train_dataset,
        ...    tokenizer = tokenizer,
        ...    kb = kb,
        ...    kb_model = kb_model,
        ...    negative_sampling_size = 10,
        ...    fit_kb_n_times = 10,
        ...    n = 1000,
        ...    top_k_size = 100,
        ...    update_top_k_every = 20,
        ...    alpha = 0.3,
        ...    seed = 42,
        ...    fit_bert = True,
        ...    fit_kb = True,
        ...    distill = True
        ... )

        >>> len(mlm_trainer.distillation.kb_logits.logits)
        1583

        >>> len(mlm_trainer.distillation.bert_logits.logits)
        72

        mlm_trainer.train()

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
        fit_bert=True,
        fit_kb=True,
        distill=True,
        seed=42,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        self.alpha = alpha

        self.kb_model = kb_model.to(self.args.device)

        self.fit_bert = fit_bert
        self.fit_kb = fit_kb
        self.fit_kb_n_times = fit_kb_n_times

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
            lr=0.00005,
        )

        self.kb = kb
        self.kb_evaluation = kb_evaluation
        self.eval_kb_every = eval_kb_every
        self.update_top_k_every = update_top_k_every

        self.step_kb = 0
        self.step_bert = 0

        self.distill = distill

        self.dataset_distillation = data.DataLoader(
            dataset=train_dataset,
            collate_fn=self.data_collator,
            batch_size=100,
        )

        self.distillation = Distillation(
            bert_model=model,
            kb_model=kb_model,
            kb=kb,
            dataset=self.dataset_distillation,
            tokenizer=tokenizer,
            entities=kb.entities,
            k=top_k_size,
            n=n,
            device=self.args.device,
        )

    @staticmethod
    def print_scores(step, name, scores):
        print("\n")
        print(f"{name} - {step}")
        for key, value in scores.items():
            print(f"\t{key}: {value:3f}")
        print("\n")

    def training_step(self, model, inputs):
        """Training step."""

        if self.fit_kb:

            for _ in range(self.fit_kb_n_times):

                self.step_kb += 1

                data = next(self.kb)
                sample, weight, mode = data["sample"], data["weight"], data["mode"]
                loss = self.link_prediction(sample=sample, weight=weight, mode=mode)

                if self.distill:
                    distillation_loss = self.distillation.distill_bert(
                        kb_model=self.kb_model,
                        sample=sample,
                    )
                    loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

                loss.backward()

                self.kb_optimizer.step()
                self.kb_optimizer.zero_grad()

                if self.kb_evaluation is not None:
                    self.link_prediction_evaluation()

                if (self.step_kb + 1) % self.update_top_k_every == 0:
                    self.distillation.update_kb(kb=self.kb, kb_model=self.kb_model)

        if self.fit_bert:

            self.step_bert += 1

            inputs.pop("mask")
            entity_ids = inputs.pop("entity_ids")
            labels = inputs["labels"]
            inputs = self._prepare_inputs(inputs)

            model.train()

            loss, outputs = self.compute_loss(model, inputs=inputs, return_outputs=True)

            if self.distill:

                distillation_loss = self.distillation.distill_transe(
                    entities=entity_ids,
                    logits=outputs.logits,
                    labels=labels,
                )

            model = model.eval()

            if self.args.n_gpu > 1:
                loss = loss.mean()
                distillation_loss = distillation_loss.mean()

            if self.distill:
                loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

            loss.backward()

            if (self.step_bert + 1) % self.update_top_k_every == 0:
                self.distillation.update_bert(
                    model=model, dataset=self.dataset_distillation, tokenizer=self.tokenizer
                )

        return loss.detach()

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

        return self
