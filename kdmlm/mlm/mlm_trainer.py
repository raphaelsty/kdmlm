import torch
import tqdm
from mkb import losses
from transformers import Trainer

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

        entities = {token: id_token for id_token, token in kb.entities.items()}

        self.mapping = {
            bert_id: entities[token]
            for token, bert_id in tokenizer.get_vocab().items()
            if token in entities
        }

        self.mask = torch.tensor(list(self.mapping.keys()))

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

        self.cosine = {
            bert_id: cos(
                kb.entity_embedding[kb_id].detach().to(device),
                kb.entity_embedding.detach()[self.mask].to(device),
            )
            for bert_id, kb_id in tqdm.tqdm(
                self.mapping.items(), desc="Cosine similarities", position=0
            )
        }

        self.kl_divergence = losses.KlDivergence()

    def distillation(self, inputs, logits):
        """Computes the knowledge distillation loss."""
        loss = 0

        # Filter on intersection
        masked_logits = logits[:, :, self.mask]

        for batch, input_id in enumerate(inputs):
            teacher = []
            student = []

            for token_id, bert_id in enumerate(input_id):

                bert_id = bert_id.item()

                if bert_id in self.cosine:
                    teacher.append(self.cosine[bert_id])
                    student.append(masked_logits[batch][token_id])

            if teacher or student:
                teacher = torch.stack(teacher, dim=0)
                student = torch.stack(student, dim=0)
                loss += self.kl_divergence(
                    student_score=student, teacher_score=teacher, T=1
                )

        return loss

    def training_step(self, model, inputs):
        """Training step."""
        model.train()

        inputs = self._prepare_inputs(inputs)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        logits = outputs.logits

        if self.args.n_gpu > 1:
            loss = loss.mean()

        loss = (1 - self.alpha) * loss + self.alpha * self.distillation(
            inputs=inputs["input_ids"], logits=logits
        )

        loss.backward()

        return loss.detach()
