import torch
from torch.utils.data import Dataset

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
    >>> from torch.utils.data import DataLoader
    >>> from transformers import BertTokenizer

    >>> from mkb import datasets as mkb_datasets

    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    >>> entities = mkb_datasets.Fb15k237(1, pre_compute=False).entities

    >>> dataset = datasets.KDDataset(
    ...     dataset=datasets.LoadFromFolder(folder=folder),
    ...     tokenizer=tokenizer,
    ...     entities=entities,
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

    >>> assert x['input_ids'][0].shape[0] ==  x['mask'][0].shape[0]
    >>> assert x['labels'][0].shape[0] == x['mask'][0].shape[0]

    >>> tokenizer.decode(x['input_ids'][0])
    '[CLS] lincoln is believed to have had depression ( mood ), [MASK], and malaria. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

    >>> tokenizer.decode(x['input_ids'][1])
    '[CLS] kennedy / a >, and [MASK] [MASK] have been the top - ranked presidents in eight surveys, according to gallup. [SEP]'

    >>> x['entity_ids']
    tensor([[10908],
            [10403]])

    >>> entities = {value: key for key, value in entities.items()}

    >>> entities[10908]
    'smallpox'

    >>> entities[10403]
    'Ronald Reagan'

    """

    def __init__(self, dataset, tokenizer, entities, n_masks=None, sep="|"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.entities = entities
        self.sep = sep
        self.n_masks = n_masks
        self.mask_id = self.tokenizer.encode(self.tokenizer.mask_token, add_special_tokens=False)[0]

    def __getitem__(self, idx):
        sentence, entity = self.dataset[idx]
        entity_id = self.entities[entity]
        input_ids = self.tokenizer.encode(sentence)
        data = self.get_mask_labels_ids(
            sentence=self.tokenizer.tokenize(sentence), input_ids=input_ids, n_masks=self.n_masks
        )
        data["entity_ids"] = torch.tensor([entity_id])
        return data

    def __len__(self):
        return self.dataset.__len__()

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
        ...     entities=mkb_datasets.Fb15k237(1, pre_compute=False).entities,
        ...     sep='|'
        ... )

        >>> sentence = '|Renault Zoe| cars are fun to drive on |French| roads.'

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

        """
        mask, labels = [], []
        stop, entities = False, False
        stop_label, label = False, False
        ids = []
        n_masked = 0

        sentence.insert(0, "[CLS]")
        sentence.append("[SEP]")

        for token, input_id in zip(sentence, input_ids):

            if token == self.sep:

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
                                ids.append(self.mask_id)
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
