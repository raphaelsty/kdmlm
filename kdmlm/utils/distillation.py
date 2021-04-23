import collections

import torch

__all__ = [
    "bert_top_k",
    "distillation_index",
    "expand_bert_logits",
    "get_tensor_distillation",
    "index",
    "mapping_entities",
]


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

        >>> tokenizer.decode([entities_bert[0]])
        'dominican'

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
    kb_entities = torch.tensor(list(mapping_kb_bert.keys()), dtype=torch.int64)
    bert_entities = torch.tensor(list(mapping_kb_bert.values()), dtype=torch.int64)
    return kb_entities, bert_entities


def get_tensor_distillation(kb_entities):
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
        [torch.tensor([0 for _ in range(len(kb_entities))], dtype=torch.int64) for _ in range(3)],
        dim=1,
    )
    tails = heads.clone()
    heads[:, 0] = torch.tensor([e for e in kb_entities], dtype=torch.int64)
    tails[:, 2] = torch.tensor([e for e in kb_entities], dtype=torch.int64)
    return heads, tails


def expand_bert_logits(logits, labels, bert_entities):
    """Filter bert logits on labels. Extract only logits which have label != 100.
    Then expand the logits to match with kb outputs logits.

    Parameters:
    -----------
        logits (torch.tensor): Output MLM logits of bert.
        labels (torch.tensor): Ids of the tokens to retrieve.
        bert_entities (torch.tensor): Tensor of ids of entities.

    Example:
    --------

        >>> from kdmlm import utils
        >>> import torch

        >>> logits = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.2, 0.8]])
        >>> labels = torch.tensor([-100, 1])
        >>> bert_entities = torch.tensor([0, 1, 0, 2])

        >>> utils.expand_bert_logits(logits, labels, bert_entities)
        tensor([[0.2000, 0.2000, 0.2000, 0.8000]])

    """
    mask_labels = labels != -100
    logits = logits[mask_labels]
    logits = torch.index_select(logits, 1, bert_entities)
    return logits


def index(tokenizer, entities, window_size=2):
    """Map entities with tokenizer.

    Parameters:

        entities (dict): Entities of the knowledge base.

    """
    index = []

    for key, value in entities.items():

        ids = tokenizer.encode(key, add_special_tokens=False)[:window_size]

        # [unused0]
        if len(ids) < window_size:
            ids.append(1)

        index.append(torch.tensor(ids))

    return torch.stack(index, dim=0)


def mapping_entities(tokenizer, max_tokens, entities):
    """Allows to compute P(entities | context).

    An entity is made of multiples tokens, i.e Buzz Aldrin is understood as:
    ['buzz', 'al', '##dr', '##in'].

    Parameters
    ----------
        tokenizer: HuggingFace Tokenizer.
        max_tokens: Number of sub-word unit to consider to compute the probability of an entity.
        entities: Entities of the knowledge base.

    Examples
    --------

    >>> from kdmlm import utils
    >>> from mkb import datasets as mkb_datasets
    >>> from transformers import DistilBertTokenizer

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    >>> tokens, order = utils.mapping_entities(
    ...     tokenizer=tokenizer, max_tokens=15, entities=kb.entities)

    >>> len(tokens.keys())
    15

    >>> len(tokens[10].keys())
    10

    >>> len(tokens[1].keys())
    1

    >>> len(tokens[16].keys())
    0

    >>> tokenizer.decode([tokens[1][0][1]])
    'anonymous'

    >>> tokenizer.decode([tokens[2][0][1], tokens[2][1][1]])
    'drama film'

    >>> assert len(order) == len(kb.entities)

    """

    tokens = collections.defaultdict(lambda: collections.defaultdict(list))
    indexes = collections.defaultdict(list)

    for key, value in entities.items():

        token = tokenizer.encode(key, add_special_tokens=False)
        n_tokens = min(len(token), max_tokens)
        indexes[n_tokens].append(value)

        for i, token_id in enumerate(token):
            if i >= max_tokens:
                break
            tokens[n_tokens][i].append(token_id)

    order = []

    for i in range(1, max_tokens + 1):
        order += indexes[i]
    order = torch.tensor(order)

    return tokens, order


def bert_top_k(logits, tokens, order, max_tokens, k):
    """Retrieve entities depending on the context.

    Parameters
    ----------
        logits (torch.tensor): Output logits of bert.
        tokens (dict): Mapping of entities between the tokenizer and the knowledge base.
        order (torch.tensor): Mapping between candidates and original order

    Examples
    --------

    >>> from kdmlm import utils
    >>> from kdmlm import datasets
    >>> from mkb import datasets as mkb_datasets

    >>> from transformers import DistilBertTokenizer
    >>> from transformers import DistilBertForMaskedLM

    >>> import torch
    >>> from torch.utils.data import DataLoader

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    >>> kb = mkb_datasets.Fb15k237(1, pre_compute=False)

    >>> tokens, order = utils.mapping_entities(
    ...     tokenizer=tokenizer, max_tokens=15, entities=kb.entities)

    >>> dataset = DataLoader(
    ...     dataset = datasets.KDDataset(
    ...         dataset = datasets.Sample(),
    ...         tokenizer = tokenizer,
    ...         n_masks = None,
    ...     ),
    ...     collate_fn = datasets.Collator(tokenizer=tokenizer),
    ...     batch_size = 3,
    ... )

    >>> for sample in dataset:
    ...     break

    >>> with torch.no_grad():
    ...     outputs = model(
    ...         input_ids = sample["input_ids"],
    ...         attention_mask = sample["attention_mask"]
    ...     )
    ...     logits = outputs.logits[sample["labels"] != -100]

    >>> scores, candidates = utils.bert_top_k(
    ...     logits=logits, tokens=tokens, order=order, max_tokens=15, k=10)

    >>> e = {value: key for key, value in kb.entities.items()}

    >>> tokenizer.decode(sample["input_ids"][0])
    '[CLS] realizing clay was unlikely to win the presidency, he supported general [MASK] [MASK] for the whig nomination in the a [SEP] [PAD]'

    >>> e[sample["entity_ids"][0].item()]
    'Zachary Taylor'

    >>> for s, c in zip(scores[0], candidates[0]):
    ...     print(f"{e[c.item()]}: {s:2f}")
    John Abraham: 6.836702
    Richard Benjamin: 6.816127
    Henry James: 6.808833
    William James: 6.785437
    John Oliver: 6.747335
    Benjamin Franklin: 6.655079
    Andrew Jackson: 6.490297
    Douglas Adams: 6.213402
    John Adams: 6.156914
    Jackson: 6.075762

    """
    scores = []

    for i in range(1, max_tokens + 1):
        index = torch.tensor(tokens[i][0])
        p_e_c = torch.index_select(logits, 1, index) / i

        for j in range(1, i):
            index = torch.tensor(tokens[i][j])
            p_e_c += torch.index_select(logits, 1, index) / i
        scores.append(p_e_c)

    scores = torch.cat(scores, dim=-1)
    scores, indices = torch.topk(scores, k)
    candidates = torch.stack([torch.index_select(order, 0, idx) for idx in indices])
    return scores, candidates
