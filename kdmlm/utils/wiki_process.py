import collections
import json
import os
import random
import re
from urllib import parse

import tqdm

__all__ = ["WikiProcess"]


class WikiProcess:
    """Pipeline to clean wikipedia text.

    Wikipedias data are available following ERNIE pre-processing dataset procedure using:
    git clone https://github.com/thunlp/ERNIE.git
    cd ERNIE

    Download Wikidump (warning it's about 19 GB):
    wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

    Read wikidump:
    python3 pretrain_data/WikiExtractor.py enwiki-latest-pages-articles.xml.bz2 -o pretrain_data/output -l --min_text_length 100 --filter_disambig_pages -it abbr,b,big --processes 4

    Parameters:
    -----------
        path_folder: Data path directory.

    References:
    1. [ERNIE: Enhanced Language Representation with Informative Entities](https://github.com/thunlp/ERNIE)

    Example:

    >>> import collections

    >>> from mkb import datasets
    >>> from kdmlm import utils

    Consider only entities that are a tail at least once in the KB.
    >>> kb = datasets.Fb15k237(1, pre_compute=False)
    >>> filter = utils.filter_entities(train=kb.train_triples)

    >>> filter.keys()
    dict_keys(['frequencies', 'tail', 'head'])

    >>> entities = {
    ...    e: id_e for e, id_e in kb.entities.items()
    ...    if id_e in filter["head"] or id_e in filter["tail"]
    ... }

    >>> wiki = utils.WikiProcess(
    ...    path_folder = 'pretrain_data/output',
    ...    entities = entities,
    ... )

    for file in wiki:
        print(file)
        break

    """

    def __init__(self, path_folder, pad=512, sep="|", entities={}):

        self.path_folder = path_folder
        self.pad = pad
        self.sep = sep
        self.entities = entities

        self.replacements = [
            ("\n \n", ""),
            ("\n", ""),
            ("'", ""),
            ("<doc> ", ""),
            (self.sep.replace(" ", ""), ""),
            (self.sep, " "),
        ]

    def read(self):
        """Read files."""
        for folder in os.listdir(self.path_folder):
            for file in os.listdir(os.path.join(self.path_folder, folder)):
                path_file = os.path.join(self.path_folder, folder, file)
                try:
                    with open(path_file) as f:
                        yield f.readlines()
                except:
                    pass

    def join_file(self):
        """Join list of files."""
        yield from (" ".join(file) for file in self.read())

    def split_doc(self):
        """Split documents."""
        for file in self.join_file():
            yield from file.split("</doc>")[:-1]

    def remove_doc_id(self):
        """Remove doc id from dataset."""
        yield from (re.sub("(?<=<doc)(.*?)(?=>)", "", file) for file in self.split_doc())

    def make_replacements(self):
        """Clean documents."""
        for file in self.remove_doc_id():
            for key, value in self.replacements:
                file = file.replace(key, value)
            yield file

    def sentence_segmentation(self):
        """Tokenize sentences."""
        for file in self.make_replacements():
            for sentence in file.split(". "):
                yield sentence + "."

    def pad_sentence(self):
        """Pad sentences with selected maximum model length."""
        for sentence in self.make_replacements():
            sentence = sentence[: self.pad]
            sentence = ".".join(sentence.split(".")[:-1]) + "."
            yield sentence

    def clean_entities(self):
        """Find entities between <a href> </a> html balises and decode them."""
        for sentence in self.pad_sentence():

            found = False
            entities = []

            for entity in re.findall('(?<=<a href=")(.*?)(?=</a>)', sentence):
                try:
                    url, id_e = entity.split('">')
                except:
                    continue
                clean_url = parse.unquote(url)
                if clean_url in self.entities:
                    sentence = sentence.replace(
                        f'<a href="{url}">{id_e}</a>',
                        f"{self.sep} {id_e} {self.sep}",
                    )

                    found = True
                    entities.append(clean_url)
                else:
                    sentence = sentence.replace(f'<a href="{url}">{id_e}</a>', f"{clean_url}")
            if found:
                # Some sentence may have been padded on an url. Remove padded url.
                sentence = re.sub(r"<|href\S+", "", sentence)
                yield sentence, entities

    def __iter__(self):
        """Yield cleaned document."""
        yield from self.clean_entities()

    def export(self, folder, size):
        """Export sentences to txt file in the selected folder."""
        n_iter = 0
        batch = []
        filename = 0
        for sentence in tqdm.tqdm(self, position=0, desc="Exporting sentences."):
            n_iter += 1
            batch.append(sentence)
            if n_iter == size:
                # Export processed sentences.
                open(os.path.join(folder, f"{filename}.txt"), "w").write(" \n".join(batch))
                filename += 1
                n_iter = 0
                batch = []

    def fb15k237one(self, folder, size, tokenizer):
        """Wikipedia dumps with mentions that are part of the vocabulary of Bert."""
        n_iter = 0
        batch = []
        filename = 0

        entities_found = collections.defaultdict(int)

        for sentence, entities in tqdm.tqdm(self, position=0, desc="Exporting sentences."):

            # Select random entities as candidates
            entities_order = [_ for _ in range(len(entities))]
            random.shuffle(entities_order)
            found = False

            # Check if there at least one entities that match the condition:
            # Single token mention.
            for e in entities_order:
                e_pos = 0

                for i, mention in enumerate(sentence.split("|")):

                    if i % 2:

                        if e_pos == e and len(tokenizer.tokenize(mention)) == 1:
                            found = True
                            new_sentence = sentence.split("|")
                            new_sentence[i] = f"| {new_sentence[i]} |"
                            # Remove unicodes caracters:
                            new_sentence = (
                                "".join(new_sentence).strip().replace("  ", " ").replace("\\", "")
                            )
                            new_sentence = new_sentence.encode("ascii", "ignore").decode()

                            entity = entities[e]
                            entities_found[entity] += 1

                        e_pos += 1

                if found:
                    break

            if found:
                n_iter += 1
                batch.append({"sentence": new_sentence, "entity": entity})

            if n_iter == size:
                # Export processed sentences.
                with open(os.path.join(folder, f"{filename}.json"), "w") as fp:
                    json.dump(batch, fp)
                filename += 1
                n_iter = 0
                batch = []

        with open(f"{folder}_entities.json", "w") as fp:
            json.dump(entities_found, fp)
