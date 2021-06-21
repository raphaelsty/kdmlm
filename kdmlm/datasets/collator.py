import torch
from transformers import DataCollatorForLanguageModeling

__all__ = ["Collator"]


class Collator(DataCollatorForLanguageModeling):
    """Format output of KDDataset. _collate_fn method of KDDataset."""

    def __init__(self, tokenizer, mlm=True, mlm_probability=0.1):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

    def __call__(self, examples):
        """Pad input examples and replace masked tokens with [MASK]."""
        batch = {
            "input_ids": _collate_batch(
                [x["input_ids"] for x in examples],
                self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
            "labels": _collate_batch(
                [x["labels"] for x in examples],
                self.tokenizer,
                pad_token_id=-100,  # -100 correspond to labels that we de not compute.
            ),
            "mask": _collate_batch(
                [x["mask"] for x in examples],
                self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id,
            ).bool(),
        }

        if "entity_ids" in examples[0]:

            batch["entity_ids"] = torch.stack([x["entity_ids"] for x in examples])

        batch["input_ids"], batch["labels"] = self.replace_tokens(
            inputs=batch["input_ids"],
            mask=batch["mask"],
            labels=batch["labels"],
        )

        batch["attention_mask"] = batch["input_ids"] != self.tokenizer.pad_token_id
        return batch

    def replace_tokens(self, inputs, mask, labels):
        """Replace masked tokens with a certain probability entities."""
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & mask
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, labels


def _collate_batch(examples, tokenizer, pad_token_id):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    result = examples[0].new_full([len(examples), max_length], pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result
