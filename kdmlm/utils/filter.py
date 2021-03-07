import collections

__all__ = ["filter_entities"]


def filter_entities(train) -> dict:
    """Extract entities of an input knowledge base that are a tail at least once in the
    training set.

    Parameters:
    -----------
        train (list): List of training triplets.

    """
    filtered_entities = collections.defaultdict(list)
    for head, relation, tail in train:
        filtered_entities[tail].append((head, relation, tail))
    return filtered_entities
