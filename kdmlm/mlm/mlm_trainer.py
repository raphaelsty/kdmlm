import random

import torch
from mkb import evaluation as mkb_evaluation
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
        device: cuda / cpu.
        alpha (float): Weight to give to knowledge distillation.


    Example:

        >>> from kdmlm import mlm
        >>> from mkb import datasets
        >>> from mkb import models

        >>> from transformers import BertTokenizer
        >>> from transformers import BertForMaskedLM

        >>> from transformers import DataCollatorForLanguageModeling
        >>> from transformers import LineByLineTextDataset
        >>> from transformers import TrainingArguments

        >>> import torch
        >>> _ = torch.manual_seed(42)

        >>> import pathlib
        >>> path_data = pathlib.Path(__file__).parent.joinpath('./../datasets/rugby.txt')

        >>> device = 'cpu'

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = BertForMaskedLM.from_pretrained('bert-base-uncased')

        >>> dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=path_data,
        ...    block_size=512)

        >>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
        ...    mlm_probability=0.15)

        >>> training_args = TrainingArguments(output_dir = f'./checkpoints',
        ... overwrite_output_dir = True, num_train_epochs = 10,
        ... per_device_train_batch_size = 64, save_steps = 500, save_total_limit = 1,
        ... do_train = True,  do_predict = True)

        >>> kb = datasets.Wn18rrText(1, pre_compute = False)

        >>> kb = models.TransE(entities = kb.entities, relations = kb.relations,
        ...    hidden_dim = 200, gamma = 8)

        >>> kb.entities = {key: value.split('.')[0] for key, value in kb.entities.items()}

        >>> entities = {value: key for key, value in kb.entities.items()}

        >>> mlm_trainer = MlmTrainer(model=model, args=training_args,
        ...    data_collator=data_collator, train_dataset=dataset, tokenizer=tokenizer,
        ...    kb=kb, device=device, alpha = 0.5)

        >>> mlm_trainer.train()
        {'train_runtime': 85.9931, 'train_samples_per_second': 0.116, 'epoch': 10.0}
        TrainOutput(global_step=10, training_loss=0.8714811325073242, metrics={'train_runtime': 85.9931, 'train_samples_per_second': 0.116, 'epoch': 10.0})

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
        device="cuda",
        alpha=0.5,
        seed=42,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        self.device = device
        self.alpha = alpha

        # Filter entities that are part of a tail
        self.triples = filter_entities(train=kb.train)
        self.kb_model = kb_model

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
        self.entities_kb = torch.Tensor(list(mapping_kb_bert.keys()))
        self.entities_bert = torch.Tensor(list(mapping_kb_bert.values()))

        self.tensor_distillation = torch.stack(
            [
                torch.Tensor([0 for _ in range(len(self.entities_kb))]),
                torch.Tensor([0 for _ in range(len(self.entities_kb))]),
                torch.Tensor([e for e in self.entities_kb]),  # Distill only tails
            ],
            dim=1,
        )

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
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.00005,
        )

    def distillation(self, logits, kb_logits, labels, student=True):
        """Computes the knowledge distillation loss."""
        loss = 0
        mask_labels = labels != -100
        logits = logits[mask_labels]
        logits = torch.index_select(logits, 1, self.entities_bert)

        if student:

            loss += self.kl_divergence(
                student_score=logits, teacher_score=kb_logits, T=1
            )

        else:

            loss += self.kl_divergence(
                student_score=kb_logits, teacher_score=logits, T=1
            )

        return loss

    def training_step(self, model, inputs):
        """Training step."""
        model.train()

        inputs = self._prepare_inputs(inputs)

        sample_distillation = self.get_sample_tensor_distillation(
            list_tails=inputs["entity_ids"]
        )
        sample, distillation = (
            sample_distillation["sample"],
            sample_distillation["distillation"],
        )

        student = self.is_student()

        if student:
            model.train()
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            with torch.no_grad():
                kb_logits = self.kb_model(distillation)

        else:
            model = model.eval()
            with torch.no_grad():
                outputs = model(inputs["input_ids"])

            loss = self.link_prediction(sample=sample)
            kb_logits = self.kb_model(distillation)

        distillation_loss = self.alpha * self.distillation(
            logits=outputs.logits,
            kb_logits=kb_logits,
            labels=inputs["labels"],
            student=student,
        )

        if self.args.n_gpu > 1:
            loss = loss.mean()
            distillation_loss = distillation_loss.mean()

        loss = (1 - self.alpha) * loss + distillation_loss

        loss.backward()

        return loss.detach()

    def get_sample_tensor_distillation(self, list_tails):
        """Init tensor to distill the knowledge of the KB. Also return the input sample
        for the link prediction task.
        """
        sample = []
        distillation = []

        for tail in list_tails:
            head, relation, _ = random.choice(self.triples[tail.item()])
            new_tensor = self.tensor_distillation.clone()
            new_tensor[:, 0] = head
            new_tensor[:, 0] = relation

            distillation.append(new_tensor)
            sample.append(torch.Tensor([[head, relation, tail]]))

        return {
            "sample": torch.stack(sample),
            "distillation": torch.stack(distillation),
        }

    def is_student(self):
        """Wether to train bert or transe."""
        self.n_student += 1
        if self.n_student % 2 == 0:
            return False
        else:
            return True

    def link_prediction(self, sample):
        """"Method dedicated to link prediction task."""
        self.n_mode += 1
        if self.n_mode % 2 == 0:
            mode = "head-batch"
        else:
            mode = "tail-batch"

        negative_sample = self.negative_sampling.generate(sample=sample, mode=mode).to(
            self.device
        )
        positive_score = self.kb_model(sample)
        negative_score = self.kb_model(
            sample=sample, negative_sample=negative_sample, mode=mode
        )
        loss = self.mkb_losses(
            positive_score, negative_score, weight=1
        )  # TODO: Add custom weights.
        return loss
