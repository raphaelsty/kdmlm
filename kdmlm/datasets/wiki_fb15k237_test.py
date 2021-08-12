import json
import pathlib

__all__ = ["WikiFb15k237Test"]


class WikiFb15k237Test:
    """Wikipedia sample to evaluate perplexity.

    Example
    -------

    >>> from kdmlm import datasets
    >>> test_dataset = datasets.WikiFb15k237Test()

    >>> for sentence in test_dataset:
    ...     break

    >>> sentence
    'Doki Doki Morning The song introduced all three members to heavy metal music; Suzuka Nakamoto commenting how she had never heard such musical heaviness before, while Yui Mizuno initially had more interest in dancing to the music rather than singing. During song production, the signature Kitsune hand gesture (similar to the sign of the horns) was formed.'

    """

    def __init__(self):
        with open(
            pathlib.Path(__file__).parent.joinpath("wiki_fb15k237_test/0.json"),
            encoding="utf-8",
        ) as input_file:
            self.dataset = json.load(input_file)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for document in self.dataset:
            yield document["sentence"].replace(" |", "")
