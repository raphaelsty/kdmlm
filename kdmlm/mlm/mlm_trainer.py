import torch
import tqdm
from mkb import losses
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
        train_triplets,
        device="cuda",
        alpha=0.5,
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
        self.entities = filter_entities(train=train_triplets)

        entities_to_bert = {
            id_e: tokenizer.decode([tokenizer.encode(e, add_special_tokens=False)[0]])
            for e, id_e in self.entities.items()
        }

        mapping_kb_bert = {
            id_e: tokenizer.encode(e, add_special_tokens=False)[0]
            for id_e, e in entities_to_bert.items()
        }

        # Entities ID of the knowledge base Kb and Bert ordered.
        self.entities_kb = torch.tensor(list(mapping_kb_bert.keys()))
        self.entities_bert = torch.tensor(list(mapping_kb_bert.values()))

        self.kl_divergence = losses.KlDivergence()

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

        kb_logits = None

        student = True

        if student:

            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        else:

            pass

        if self.args.n_gpu > 1:
            loss = loss.mean()

        distillation_loss = self.alpha * self.distillation(
            logits=outputs.logits,
            kb_logits=kb_logits,
            labels=inputs["labels"],
            student=student,
        )

        loss = (1 - self.alpha) * loss + distillation_loss

        loss.backward()

        return loss.detach()
