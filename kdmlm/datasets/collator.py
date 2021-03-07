import torch
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import BatchEncoding

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

        batch["input_ids"], batch["labels"] = self.replace_tokens(
            inputs=batch["input_ids"],
            mask=batch["mask"],
            labels=batch["labels"],
        )
        return batch

    def replace_tokens(self, inputs, mask, labels):
        """Replace masked tokens with a certain probability entities."""
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # TODO REPLACE ONLY WITH RANDOM ENTITIES
        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & mask
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
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
