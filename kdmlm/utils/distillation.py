import torch

__all__ = ["distillation_index", "get_tensor_distillation"]


def distillation_index(tokenizer, entities):
    """Return ids of entities in the KB and in Bert.
    Useful to expand logits of berts.

    Parameters:
    -----------
        tokenizer: HuggingFace tokenizer.
        entities (dict): Mapping between labels of entities and their ids.

    Example:
    --------

        >>> from mkb import datasets
        >>> from transformers import BertTokenizer
        >>> from kdmlm import utils

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        >>> dataset = datasets.Fb15k237(1, pre_compute=False)
        >>> entities_kb, entities_bert = utils.distillation_index(
        ...    tokenizer=tokenizer,
        ...    entities=dataset.entities
        ... )

        >>> tokenizer.decode(entities_bert[0])
        'd o m i n i c a n'

        >>> list(dataset.entities.keys())[entities_kb[0]]
        'Dominican Republic'

    """
    entities_to_bert = {
        id_e: tokenizer.decode([tokenizer.encode(e, add_special_tokens=False)[0]])
        for e, id_e in entities.items()
    }
    mapping_kb_bert = {
        id_e: tokenizer.encode(e, add_special_tokens=False)[0]
        for id_e, e in entities_to_bert.items()
    }
    tensor_entities_kb = torch.tensor(list(mapping_kb_bert.keys()), dtype=torch.int64)
    tensor_entities_bert = torch.tensor(
        list(mapping_kb_bert.values()), dtype=torch.int64
    )
    return tensor_entities_kb, tensor_entities_bert


def get_tensor_distillation(tensor_entities):
    """Get tensors dedicated to distillation.

    Example:
    --------

        >>> from kdmlm import utils
        >>> heads, tails = utils.get_tensor_distillation([1, 2, 3])

        >>> heads
        tensor([[1, 0, 0],
                [2, 0, 0],
                [3, 0, 0]])

        >>> tails
        tensor([[0, 0, 1],
                [0, 0, 2],
                [0, 0, 3]])

    """
    heads = torch.stack(
        [
            torch.tensor([0 for _ in range(len(tensor_entities))], dtype=torch.int64)
            for _ in range(3)
        ],
        dim=1,
    )
    tails = heads.clone()
    heads[:, 0] = torch.tensor([e for e in tensor_entities], dtype=torch.int64)
    tails[:, 2] = torch.tensor([e for e in tensor_entities], dtype=torch.int64)
    return heads, tails
