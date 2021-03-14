import collections

__all__ = ["filter_entities"]


def filter_entities(train):
    """Extract entities of an input knowledge base that are a tail at least once in the
    training set.

    Parameters:
    -----------
        train (list): List of training triplets.

    Example:
    --------

    >>> from kdmlm import utils
    >>> train = [(1, 2, 3), (2, 2, 3), (1, 2, 4)]
    >>> utils.filter_entities(train)
    {'frequencies': defaultdict(<class 'int'>, {1: 2, 3: 2, 2: 1, 4: 1}), 'tail': {3: [(1, 2, 3), (2, 2, 3)], 4: [(1, 2, 4)]}, 'head': {1: [(1, 2, 3), (1, 2, 4)], 2: [(2, 2, 3)]}}

    """
    tails = {}
    heads = {}
    frequencies = collections.defaultdict(int)

    for head, relation, tail in train:

        if head not in heads:
            heads[head] = []

        if tail not in tails:
            tails[tail] = []

        tails[tail].append((head, relation, tail))
        heads[head].append((head, relation, tail))

        frequencies[head] += 1
        frequencies[tail] += 1

    return {"frequencies": frequencies, "tail": tails, "head": heads}
