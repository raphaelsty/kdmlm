import torch
from torch.utils.data import Dataset

from . import Collator

___all__ = ["KDDataset"]


class KDDataset(Dataset):
    """Load and access textual data. KDDataset mask the first entity in the input
    sentence.

    Arguments:
    ----------
        dataset (kdmlm.datasets.LoadDatasetFromFile)
        tokenizer: Bert tokenizer.
        sep (str): Separator to identify entities.
        entities (dict): Mapping between entities names and entities ids.

    Example:
    --------

    >>> import pathlib

    >>> from pprint import pprint

    >>> from kdmlm import datasets
    >>> from mkb import datasets as mkb_datasets

    >>> from torch.utils.data import DataLoader

    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    >>> entities = mkb_datasets.Fb15k237(1, pre_compute=False).entities

    >>> dataset = datasets.KDDataset(
    ...     dataset=datasets.LoadFromFolder(folder=folder, entities=entities, tokenizer=tokenizer),
    ...     tokenizer=tokenizer,
    ...     sep='|'
    ... )

    >>> data_loader = DataLoader(
    ...    dataset,
    ...    batch_size=2,
    ...    collate_fn=datasets.Collator(tokenizer=tokenizer),
    ... )

    >>> for x in data_loader:
    ...    break

    >>> assert x['input_ids'][0].shape[0] == x['input_ids'][1].shape[0]
    >>> assert x['mask'][0].shape[0] == x['mask'][1].shape[0]
    >>> assert x['labels'][0].shape[0] == x['labels'][1].shape[0]
    >>> assert x['attention_mask'][0].shape[0] == x['attention_mask'][1].shape[0]

    >>> assert x['input_ids'][0].shape[0] ==  x['mask'][0].shape[0]
    >>> assert x['labels'][0].shape[0] == x['mask'][0].shape[0]
    >>> assert x['mask'][0].shape[0] == x['attention_mask'][0].shape[0]

    >>> tokenizer.decode(x['input_ids'][0])
    '[CLS] realizing clay was unlikely to win the presidency, he supported general [MASK] [MASK] for the whig nomination in the a [SEP]'

    >>> tokenizer.decode(x['input_ids'][1])
    '[CLS] the democrats nominated former secretary of state [MASK] [MASK] and the know - nothings nominated former whig president a [SEP] [PAD]'

    >>> x['entity_ids']
    tensor([[11839],
            [11190]])

    >>> entities = {value: key for key, value in entities.items()}

    >>> entities[11839]
    'Zachary Taylor'

    >>> entities[11190]
    'James Buchanan'

    """

    def __init__(self, dataset, tokenizer, n_masks=None, sep="|", mlm_probability=0):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sep = sep
        self.n_masks = n_masks
        self.collator = Collator(tokenizer=tokenizer)
        self.mlm_probability = mlm_probability

    def __getitem__(self, idx):
        sentence, entity_id = self.dataset[idx]

        if self.mlm_probability < torch.rand(1).item():

            input_ids = self.tokenizer.encode(sentence)

            data = self.get_mask_labels_ids(
                sentence=self.tokenizer.tokenize(sentence),
                input_ids=input_ids,
                n_masks=self.n_masks,
            )

            data["entity_ids"] = torch.tensor([entity_id])

        else:

            sentence = sentence.replace(self.sep, "")
            sentence = sentence.replace("  ", " ")
            input_ids = self.tokenizer.encode(sentence)

            data = self.get_mlm_masking(
                sentence=self.tokenizer.tokenize(sentence),
                input_ids=input_ids,
            )

        return data

    def __len__(self):
        return self.dataset.__len__()

    def get_mlm_masking(self, sentence, input_ids):
        """Default mlm masking


        Example:
        --------
        >>> import pathlib

        >>> from pprint import pprint

        >>> from kdmlm import datasets
        >>> from mkb import datasets as mkb_datasets

        >>> import torch
        >>> from torch.utils.data import DataLoader

        >>> _ = torch.manual_seed(42)

        >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

        >>> from transformers import BertTokenizer
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        >>> entities = mkb_datasets.Fb15k237(1, pre_compute=False).entities

        >>> dataset = datasets.KDDataset(
        ...     dataset=datasets.LoadFromFolder(folder=folder,
        ...         entities=entities,
        ...         tokenizer=tokenizer
        ...     ),
        ...     tokenizer=tokenizer,
        ...     sep='|',
        ...     mlm_probability=1,
        ... )

        >>> data_loader = DataLoader(
        ...    dataset,
        ...    batch_size=2,
        ...    collate_fn=datasets.Collator(tokenizer=tokenizer),
        ... )

        >>> for data in data_loader:
        ...     break

        >>> for id, input_id in enumerate(data["input_ids"].tolist()[0]):
        ...     if input_id == tokenizer.mask_token_id:
        ...         break

        >>> data["mask"].tolist()[0][id]
        True

        >>> list(data.keys())
        ['input_ids', 'labels', 'mask', 'attention_mask']

        """
        sentence.insert(0, self.tokenizer.cls_token)
        sentence.append(self.tokenizer.sep_token)
        mask_id = torch.randint(low=1, high=len(sentence) - 2, size=(1,)).item()
        mask = [False if i != mask_id else True for i in range(len(input_ids))]
        labels = [-100 if i != mask_id else input_id for i, input_id in enumerate(input_ids)]
        ids = [
            input_id if i != mask_id else self.tokenizer.mask_token_id
            for i, input_id in enumerate(input_ids)
        ]

        return {"mask": mask, "labels": labels, "input_ids": ids}

    def get_mask_labels_ids(self, sentence, input_ids, n_masks=None):
        """Mask first entity in the sequence and return corresponding labels.
        We evaluate loss on the first part of the first entity met.

        Parameters:
        -----------
            sentence (list): Tokenized sentence.
            input_ids (list):
            n_masks (int): Fixed number of mask tokens to use to replace input entity.

        Example:
        --------

        >>> from kdmlm import datasets
        >>> from torch.utils.data import DataLoader
        >>> from transformers import BertTokenizer

        >>> from mkb import datasets as mkb_datasets

        >>> from pprint import pprint

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        >>> dataset = datasets.KDDataset(
        ...     dataset=[],
        ...     tokenizer=tokenizer,
        ...     sep='|'
        ... )

        >>> sentence = '| Renault Zoe | cars are fun to drive on | French | roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence)
        ... )

        >>> pprint(x)
        {'input_ids': [101,
                    14605,
                    11199,
                    3765,
                    2024,
                    4569,
                    2000,
                    3298,
                    2006,
                    2413,
                    4925,
                    1012,
                    102],
        'labels': [-100,
                14605,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100,
                -100],
        'mask': [False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False]}

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] renault zoe cars are fun to drive on french roads. [SEP]'


        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 1,
        ... )

        >>> pprint(x)
        {'input_ids': [101,
                14605,
                3765,
                2024,
                4569,
                2000,
                3298,
                2006,
                2413,
                4925,
                1012,
                102],
        'labels': [-100,
                    14605,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100],
        'mask': [False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False]}

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] renault cars are fun to drive on french roads. [SEP]'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 3,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] renault zoe [MASK] cars are fun to drive on french roads. [SEP]'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 5,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] renault zoe [MASK] [MASK] [MASK] cars are fun to drive on french roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> from transformers import RobertaTokenizer
        >>> tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)

        >>> dataset = datasets.KDDataset(
        ...     dataset=[],
        ...     tokenizer=tokenizer,
        ...     sep='|'
        ... )

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 5,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '<s> Renault Zoe<mask><mask><mask> cars are fun to drive on French roads.</s>'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        """
        mask, labels = [], []
        stop, entities = False, False
        stop_label, label = False, False
        ids = []
        n_masked = 0

        sentence.insert(0, self.tokenizer.cls_token)
        sentence.append(self.tokenizer.sep_token)

        for token, input_id in zip(sentence, input_ids):

            if self.sep in token:

                if not entities:
                    # Begining of an entity.
                    entities, label = True, True
                else:
                    # Ending of an entity.
                    # We will stop masking entities.
                    entities, stop = False, True

                    if n_masks is not None:
                        if n_masked < n_masks:
                            for _ in range(n_masks - n_masked):
                                ids.append(self.tokenizer.mask_token_id)
                                mask.append(True)
                                labels.append(-100)
            else:

                if stop:
                    # First entity already met.
                    entities = False

                if stop_label:
                    # First element of first entity already met.
                    label = False

                if entities:
                    n_masked += 1

                if n_masks is not None:
                    if entities and n_masked > n_masks:
                        continue

                ids.append(input_id)
                mask.append(entities)

                if label:
                    # Eval loss.
                    labels.append(input_id)
                else:
                    # Do not eval loss.
                    labels.append(-100)

                # Eval loss only for the first element of the first entity met.
                if label:
                    stop_label = True

        return {"mask": mask, "labels": labels, "input_ids": ids}

    def collate_fn(self, data):
        return self.collator(data)
