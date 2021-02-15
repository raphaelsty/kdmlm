# KbTrainer
import torch
import tqdm
from mkb import losses
from transformers import Trainer

__all__ = ["Mlm"]


class Mlm(Trainer):
    """Custom trainer to distill knowledge to bert from knowledge graphs embeddings.

    Parameters:
    -----------

        knowledge: mkb model.
    """

    def __init__(
        self, model, args, data_collator, train_dataset, tokenizer, kb, device
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        self.device = device

        self.mapping = {
            bert_id: kb.entities[token]
            for token, bert_id in tokenizer.get_vocab().items()
            if token in kb.entities
        }

        self.mask = torch.tensor(list(self.mapping.values()))

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

        loss += self.distillation(inputs=inputs, logits=logits)
        loss.backward()
        return loss.detach()
