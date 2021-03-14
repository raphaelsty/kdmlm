import pytest
import torch
from kdmlm import datasets, mlm
from mkb import datasets as mkb_datasets
from mkb import models as mkb_models
from torch.utils.data import DataLoader
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    TrainingArguments,
)

from .mlm_trainer import MlmTrainer


def test_mlm_trainer():
    """Test mlm trainer"""

    _ = torch.manual_seed(42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    dataset = datasets.KDDataset(
        dataset=["I live in |France|.", "I live in |Spain|."],
        tokenizer=tokenizer,
        entities=kb.entities,
        sep="|",
    )

    data_collator = datasets.Collator(tokenizer=tokenizer, mlm_probability=1)

    training_args = TrainingArguments(
        output_dir=f"./checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        save_steps=500,
        save_total_limit=1,
        do_train=True,
        do_predict=True,
    )

    kb_model = mkb_models.TransE(
        entities=kb.entities, relations=kb.relations, hidden_dim=20, gamma=8
    )

    data_loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=datasets.Collator(tokenizer=tokenizer),
    )

    for inputs in data_loader:
        pass

    mlm_trainer = MlmTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
        kb=kb,
        kb_model=kb_model,
        negative_sampling_size=1,
        alpha=0.5,
        seed=42,
    )

    # Test sample distillation:
    sample_mode_distillation = mlm_trainer.get_sample_mode_distillation(
        list_entities=inputs["entity_ids"]
    )
    sample, mode, distillation = (
        sample_mode_distillation["sample"],
        sample_mode_distillation["mode"],
        sample_mode_distillation["distillation"],
    )
    assert torch.equal(sample, torch.tensor([[352, 15, 12837], [6726, 87, 1959]]))
    assert mode == "head-batch"
    assert distillation.shape[0] == 2
    assert distillation.shape[1] == len(kb.entities)

    # Test Link prediction:
    loss = mlm_trainer.link_prediction(sample=sample, mode=mode)
    assert loss.item() == pytest.approx(1.162128)

    assert tokenizer.decode(inputs["input_ids"][0]) == "[CLS] i live in [MASK]. [SEP]"
    assert tokenizer.decode(inputs["input_ids"][1]) == "[CLS] i live in spain. [SEP]"
    inputs = mlm_trainer._prepare_inputs(inputs)
    assert tokenizer.decode(inputs["input_ids"][0]) == "[CLS] i live in [MASK]. [SEP]"
    assert tokenizer.decode(inputs["input_ids"][1]) == "[CLS] i live in spain. [SEP]"

    # Distillation
    inputs.pop("mask")
    inputs.pop("entity_ids")
    logits = model(inputs["input_ids"]).logits
    kb_logits = mlm_trainer.kb_model(distillation)
    student = True

    # distillation_loss = mlm_trainer.distillation(
    #    logits=logits,
    #    kb_logits=kb_logits,
    #    labels=inputs["labels"],
    #    student=student,
    # )

    # assert distillation_loss.item() == pytest.approx(0.000530, 6)

    labels = inputs["labels"]
    mask_labels = labels != -100
    logits = logits[mask_labels]
    logits = torch.index_select(logits, 1, mlm_trainer.entities_bert)

    assert logits.shape[0] == 2
    assert logits.shape[1] == len(kb.entities)

    assert kb_logits.shape[0] == 2
    assert kb_logits.shape[1] == len(kb.entities)
