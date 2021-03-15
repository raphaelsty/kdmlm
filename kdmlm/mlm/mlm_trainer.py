import pprint
import random

import torch
from mkb import losses as mkb_losses
from mkb import sampling as mkb_sampling
from transformers import Trainer

from ..utils.filter import filter_entities

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

        >>> kb = mkb_datasets.Fb15k237(1, pre_compute = False)

        >>> train_dataset = datasets.KDDataset(
        ...     dataset=datasets.Sample(),
        ...     tokenizer=tokenizer,
        ...     entities=kb.entities,
        ...     sep='|'
        ... )

        >>> data_collator = datasets.Collator(tokenizer=tokenizer)

        >>> training_args = TrainingArguments(output_dir = f'./checkpoints',
        ... overwrite_output_dir = True, num_train_epochs = 10,
        ... per_device_train_batch_size = 64, save_steps = 500, save_total_limit = 1,
        ... do_train = True,  do_predict = True)

        >>> kb_model = mkb_models.TransE(entities = kb.entities, relations = kb.relations,
        ...    hidden_dim = 1, gamma = 8)

        >>> negative_sampling_size = 2
        >>> alpha = 0.5

        >>> mlm_trainer = MlmTrainer(model=model, args=training_args,
        ...    data_collator=data_collator, train_dataset=train_dataset,
        ...    tokenizer=tokenizer, kb=kb, kb_model=kb_model,
        ...    negative_sampling_size=negative_sampling_size,
        ...    alpha=alpha, seed=42, fit_bert = True, fit_kb = True, distill = False)

        >>> mlm_trainer.train()
        {'train_runtime': 351.7002, 'train_samples_per_second': 0.057, 'epoch': 10.0}
        TrainOutput(global_step=20, training_loss=1.4457881927490235, metrics={'train_runtime': 351.7002, 'train_samples_per_second': 0.057, 'epoch': 10.0})
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
        alpha=0.5,
        seed=42,
        top_k_size=10,
        n_random_entities=10,
        fit_bert=True,
        fit_kb=True,
        distill=True,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        self.alpha = alpha

        # Filter entities that are part of a tail
        self.triples = filter_entities(train=kb.train)
        self.kb_model = kb_model.to(self.args.device)

        self.top_k_size = top_k_size
        self.n_random_entities = n_random_entities

        self.fit_bert = fit_bert
        self.fit_kb = fit_kb

        random.seed(seed)

        entities_to_bert = {
            id_e: tokenizer.decode([tokenizer.encode(e, add_special_tokens=False)[0]])
            for e, id_e in kb.entities.items()
        }

        mapping_kb_bert = {
            id_e: tokenizer.encode(e, add_special_tokens=False)[0]
            for id_e, e in entities_to_bert.items()
        }

        # Entities ID of the knowledge base Kb and Bert ordered.
        self.entities_kb = torch.tensor(
            list(mapping_kb_bert.keys()), dtype=torch.int64
        ).to(self.args.device)

        self.entities_bert = torch.tensor(
            list(mapping_kb_bert.values()), dtype=torch.int64
        ).to(self.args.device)

        self.tensor_distillation_tails = torch.stack(
            [
                torch.tensor(
                    [0 for _ in range(len(self.entities_kb))], dtype=torch.int64
                ),
                torch.tensor(
                    [0 for _ in range(len(self.entities_kb))], dtype=torch.int64
                ),
                torch.tensor(
                    [e for e in self.entities_kb], dtype=torch.int64
                ),  # Distill only tails
            ],
            dim=1,
        )

        self.tensor_distillation_heads = self.tensor_distillation_tails.clone()

        # Wether to train bert or kb model.
        self.n_student = 0
        self.n_mode = 0

        self.mkb_losses = mkb_losses.Adversarial()
        self.kl_divergence = mkb_losses.KlDivergence()

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

        self.kb_evaluation = kb_evaluation
        self.eval_kb_every = eval_kb_every
        self.step = 0
        self.distill = distill

    def filter_labels(self, logits, labels):
        """Computes the knowledge distillation loss."""
        mask_labels = labels != -100
        logits = logits[mask_labels]
        logits = torch.index_select(logits, 1, self.entities_bert)
        return logits

    def training_step(self, model, inputs):
        """Training step."""
        model.train()

        inputs.pop("mask")
        entity_ids = inputs.pop("entity_ids")
        inputs = self._prepare_inputs(inputs)

        sample_mode_distillation = self.get_sample_mode_distillation(
            list_entities=entity_ids
        )

        sample, mode, distillation = (
            sample_mode_distillation["sample"],
            sample_mode_distillation["mode"],
            sample_mode_distillation["distillation"],
        )

        student = self.is_student()

        # Fit bert or kb
        if student:
            model.train()
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        elif self.distill:
            model = model.eval()
            with torch.no_grad():
                outputs = model(inputs["input_ids"])

        if not student:
            loss = self.link_prediction(sample=sample, mode=mode)

        if self.distill:
            logits = self.filter_labels(
                outputs.logits.to(self.args.device),
                inputs["labels"].to(self.args.device),
            )

        if student and self.distill:

            with torch.no_grad():
                kb_scores = self.kb_model(distillation.to(self.args.device))
                top_k_scores = self.top_k(
                    teacher_scores=kb_scores, student_scores=logits
                )

        elif not student and self.distill:

            kb_scores = self.kb_model(distillation.to(self.args.device))
            top_k_scores = self.top_k(teacher_scores=logits, student_scores=kb_scores)

        if self.distill:

            distillation_loss = self.kl_divergence(
                student_score=top_k_scores["top_k_teacher"],
                teacher_score=top_k_scores["top_k_student"],
                T=1,
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()
            distillation_loss = distillation_loss.mean()

        if self.distill:
            loss = (1 - self.alpha) * loss + self.alpha * distillation_loss

        loss.backward()

        if not student:
            self.kb_optimizer.step()
            self.kb_optimizer.zero_grad()

        if self.kb_evaluation is not None:

            self.step += 1

            if (self.step % self.eval_kb_every) == 0:
                self.kb_evaluation.eval(
                    model=self.kb_model,
                    dataset=self.kb.valid,
                )

                self.kb_evaluation.eval(
                    model=self.kb_model,
                    dataset=self.kb.test,
                )

        return loss.detach()

    def get_sample_mode_distillation(self, list_entities):
        """Init tensor to distill the knowledge of the KB. Also return the input sample
        for the link prediction task.
        """
        sample = []
        distillation = []

        if self.n_mode % 2 == 0:
            mode = "head"
            inverse_mode = "tail"
        else:
            mode = "tail"
            inverse_mode = "head"

        self.n_mode += 1

        for entity in list_entities:
            entity = entity.item()

            if entity in self.triples[mode]:
                head, relation, tail = random.choice(self.triples[mode][entity])

            else:
                head, relation, tail = random.choice(self.triples[inverse_mode][entity])

            new_tensor = self.tensor_distillation_tails.clone()
            new_tensor[:, 0] = head
            new_tensor[:, 0] = relation

            distillation.append(new_tensor)
            sample.append(torch.tensor([head, relation, tail], dtype=torch.int64))

        return {
            "sample": torch.stack(sample),
            "distillation": torch.stack(distillation),
            "mode": f"{mode}-batch",
        }

    def is_student(self):
        """Wether to train bert or transe."""
        if self.fit_bert and not self.fit_kb:
            return True

        if not self.fit_bert:
            return False

        self.n_student += 1
        if self.n_student % 2 == 0:
            return False
        return True

    def link_prediction(self, sample, mode):
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

        loss = self.mkb_losses(
            positive_score,
            negative_score,
            weight=torch.ones(sample.shape[0]).to(self.args.device),
        )  # TODO: Add custom weights.
        return loss

    def top_k(self, teacher_scores, student_scores):
        """Returns top k scores and k' random scores."""
        top_k_index = torch.argsort(teacher_scores, dim=1, descending=True)[
            :, 0 : self.top_k_size
        ]

        random_index = torch.randint(
            len(self.entities_kb), (self.n_random_entities,)
        ).to(self.args.device)

        top_k_teacher = torch.stack(
            [
                torch.index_select(teacher_scores[i], 0, top_k_index[i])
                for i in range(teacher_scores.shape[0])
            ]
        )

        top_k_student = torch.stack(
            [
                torch.index_select(student_scores[i], 0, top_k_index[i])
                for i in range(student_scores.shape[0])
            ]
        )

        random_teacher = torch.index_select(teacher_scores, 1, random_index)
        random_student = torch.index_select(student_scores, 1, random_index)

        top_k_teacher = torch.cat((top_k_teacher, random_teacher), dim=1)
        top_k_student = torch.cat((top_k_student, random_student), dim=1)

        return {"top_k_teacher": top_k_teacher, "top_k_student": top_k_student}
