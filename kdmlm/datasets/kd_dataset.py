from torch.utils.data import Dataset

___all__ = ["KDDataset"]


class KDDataset(Dataset):
    """Load and access textual data. KDDataset mask the first entity in the input
    sentence.

    Arguments:
    ----------
        tokenizer: Bert tokenizer.
        sep (str): Separator to identify entities.

    Example:
    --------

    >>> import pathlib

    >>> from kdmlm import datasets
    >>> from torch.utils.data import DataLoader
    >>> from transformers import BertTokenizer

    >>> folder = pathlib.Path(__file__).parent.joinpath('./../datasets/sentences')

    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    >>> dataset = datasets.KDDataset(
    ...     dataset=datasets.LoadFromFolder(folder=folder),
    ...     tokenizer=tokenizer,
    ...     sep='|'
    ... )

    >>> data_loader = DataLoader(
    ...    dataset,
    ...    batch_size=2,
    ...    collate_fn=datasets.Collator(tokenizer=tokenizer),
    ... )

    >>> for x in data_loader:
    ...    print(x)
    ...    break

    """

    def __init__(self, dataset, tokenizer, sep="|"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sep = sep

    def __getitem__(self, idx):
        sentence = self.dataset[idx]
        input_ids = self.tokenizer.encode(sentence)
        return self.get_mask_labels_ids(
            sentence=self.tokenizer.tokenize(sentence), input_ids=input_ids
        )

    def __len__(self):
        return self.dataset.__len__()

    def get_mask_labels_ids(self, sentence, input_ids):
        """Mask first entity in the sequence and return corresponding labels.
        We evaluate loss on the first part of the first entity met.

        Parameters:
        -----------
            sentence (list): Tokenized sentence.

        Example:
        --------

        >>> from kdmlm import datasets
        >>> from torch.utils.data import DataLoader
        >>> from transformers import BertTokenizer

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        >>> dataset = datasets.KDDataset(
        ...     dataset=[],
        ...     tokenizer=tokenizer,
        ...     sep='|'
        ... )

        >>> sentence = '|Renault Zoe| cars are fun to drive on |French| roads.'

        >>> x = dataset.get_mask_labels_ids(
        ...    sentence = tokenizer.tokenize(sentence),
        ...    input_ids = tokenizer.encode(sentence)
        ... )

        >>> x
        {'mask': [False, True, True, False, False, False, False, False, False, False, False, False, False], 'labels': [-100, 14605, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100], 'input_ids': [101, 14605, 11199, 3765, 2024, 4569, 2000, 3298, 2006, 2413, 4925, 1012, 102]}

        >>> tokenizer.decode(x['input_ids'])
        '[CLS] renault zoe cars are fun to drive on french roads. [SEP]'

        """
        mask, labels = [], []
        stop, entities = False, False
        stop_label, label = False, False
        ids = []

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
            else:

                ids.append(input_id)

                if stop:
                    # First entity already met.
                    entities = False

                if stop_label:
                    # First element of first entity already met.
                    label = False

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
