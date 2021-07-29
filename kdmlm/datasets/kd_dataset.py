import random

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
    '[CLS] realizing clay was unlikely to win the presidency, he supported general [MASK] for the whig nomination in the a [SEP]'

    >>> tokenizer.decode(x['input_ids'][1])
    '[CLS] the democrats nominated former secretary of state [MASK] and the know - nothings nominated former whig president a [SEP] [PAD]'

    >>> x['entity_ids']
    tensor([[11839],
            [11190]])

    >>> entities = {value: key for key, value in entities.items()}

    >>> entities[11839]
    'Zachary Taylor'

    >>> entities[11190]
    'James Buchanan'

    >>> tokenizer.decode([19474])
    'zachary'

    >>> tokenizer.decode([2508])
    'james'

    >>> entities = datasets.Fb15k237One(1, pre_compute=False).entities

    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/wiki_fb15k237one')

    >>> dataset = datasets.KDDataset(
    ...     dataset=datasets.LoadFromJsonFolder(folder=folder, entities=entities),
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
    '[CLS] kenya hockey union the kenya hockey union ( khu ) is the sports governing body of field hockey in [MASK]. its headquarters are in nairobi. it is affiliated to ihf international hockey federation and ahf african hockey federation. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

    >>> tokenizer.decode(x['input_ids'][1])
    "[CLS] denis yartsev denis nikolayevich yartsev ( ; born 18 september 1990 ) is a russian [MASK] ka. he competed at the 2016 summer olympics in the judo at the 2016 summer olympics men's 73 kg event, in which he was eliminated by lasha shavdatuashvili in the repechage. [SEP]"

    """

    def __init__(
        self, dataset, tokenizer, n_masks=1, sep="|", mlm_probability=0, masking_probability=1
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sep = sep
        self.n_masks = n_masks
        self.collator = Collator(tokenizer=tokenizer)
        self.mlm_probability = mlm_probability
        self.masking_probability = masking_probability

    def __getitem__(self, idx):
        sentence, entity_id = self.dataset[idx]

        if self.mlm_probability < random.uniform(0, 1):

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

        >>> list(data.keys())
        ['input_ids', 'labels', 'mask', 'attention_mask']

        """
        sentence.insert(0, self.tokenizer.cls_token)
        sentence.append(self.tokenizer.sep_token)
        label_id = random.randint(1, len(sentence) - 2)

        if random.uniform(0, 1) > 0.15:
            mask = [False if i != label_id else True for i in range(len(input_ids))]
        else:
            mask = [False for _ in range(len(input_ids))]

        labels = [-100 if i != label_id else input_id for i, input_id in enumerate(input_ids)]

        ids = [
            input_id if i != label_id else self.tokenizer.mask_token_id
            for i, input_id in enumerate(input_ids)
        ]

        return {"mask": mask, "labels": labels, "input_ids": ids}

    def get_mask_labels_ids(self, sentence, input_ids, n_masks):
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
        ...     sep='|',
        ...     masking_probability=1,
        ... )

        >>> sentence = '| Renault | Zoe cars are fun to drive on French roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 1,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] zoe cars are fun to drive on french roads. [SEP]'

        >>> tokenizer.decode(x["labels"])
        '[UNK] renault [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] [UNK]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> sentence = '| Renault Zoe | cars are fun to drive on French roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 1,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] cars are fun to drive on french roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> sentence = '| Renault | Zoe cars are fun to drive on French roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 2,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] [MASK] zoe cars are fun to drive on french roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> sentence = '| Renault Zoe | cars are fun to drive on French roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 2,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] [MASK] cars are fun to drive on french roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> sentence = '| Renault Zoe | cars are fun to drive on | French | roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 1,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] cars are fun to drive on [MASK] roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> sentence = '| Renault Zoe | cars are fun to drive on | French | roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 2,
        ... )

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] [MASK] cars are fun to drive on [MASK] [MASK] roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        >>> sentence = '| Renault Zoe | | cars | are fun to drive on | French | roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence),
        ...    n_masks = 2,
        ... )

        >>> x["mask"]
        [False, True, True, True, True, False, False, False, False, False, True, True, False, False, False]

        >>> tokenizer.decode(x["labels"])
        '[UNK] renault [UNK] cars [UNK] [UNK] [UNK] [UNK] [UNK] [UNK] french [UNK] [UNK] [UNK] [UNK]'

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] [MASK] [MASK] [MASK] [MASK] are fun to drive on [MASK] [MASK] roads. [SEP]'

        >>> assert len(x['input_ids']) == len(x['mask']) and len(x['mask']) == len(x['labels'])

        """
        mask, labels, ids = [], [], []
        entities = False

        sentence.insert(0, self.tokenizer.cls_token)
        sentence.append(self.tokenizer.sep_token)

        for token, input_id in zip(sentence, input_ids):

            if self.sep in token:

                if not entities:
                    # Begining of an entity.
                    entities = True
                    n_masked = 0
                else:
                    # Ending of an entity.
                    # We will stop masking entities.
                    entities = False
                    # Fixed number of masks:
                    for _ in range(n_masks - n_masked):
                        ids.append(self.tokenizer.mask_token_id)
                        mask.append(True)
                        labels.append(-100)

                continue

            if entities and n_masked == 0:
                if self.masking_probability > random.uniform(0, 1):
                    ids.append(self.tokenizer.mask_token_id)
                    mask.append(True)
                else:
                    ids.append(input_id)
                    mask.append(False)
                n_masked += 1
                labels.append(input_id)

            elif not entities:
                ids.append(input_id)
                mask.append(False)
                labels.append(-100)

        return {"mask": mask, "labels": labels, "input_ids": ids}

    def collate_fn(self, data):
        return self.collator(data)
