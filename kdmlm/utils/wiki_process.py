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

    def __init__(self, tokenizer, path_folder, pad=512, sep="|", entities={}):

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

        # Mentions aims at storing all availables menttions for a given entity.
        self.mentions = collections.defaultdict(lambda: collections.defaultdict(int))
        self.id_sep = tokenizer.encode(self.sep, add_special_tokens=False)[0]

    def read(self):
        """Read files."""
        for folder in os.listdir(self.path_folder):
            if ".DS_Store" in folder:
                continue
            for file in os.listdir(os.path.join(self.path_folder, folder)):
                if ".DS_Store" in file:
                    continue
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

    def remove_title(self):
        for file in self.remove_doc_id():
            for sentence in file.split("\n"):
                if "." in sentence:
                    yield sentence

    def make_replacements(self):
        """Clean documents."""
        for file in self.remove_title():
            for key, value in self.replacements:
                file = file.replace(key, value)
            yield file

    def sentence_segmentation(self):
        """Tokenize sentences."""
        for file in self.make_replacements():
            for sentence in file:
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
                    url, mention = entity.split('">')
                except:
                    continue

                clean_url = parse.unquote(url)

                if clean_url in self.entities:

                    # Replace url by the mention of the entity
                    sentence = sentence.replace(
                        f'<a href="{url}">{mention}</a>',
                        f"{self.sep} {mention} {self.sep}",
                    )

                    # if mention not in self.mentions[clean_url]:
                    # Store existing mention for a given entity.
                    self.mentions[clean_url][mention] += 1

                    found = True
                    entities.append(clean_url)
                else:
                    # Remove existing url even if we do not use it..
                    sentence = sentence.replace(f'<a href="{url}">{mention}</a>', f"{clean_url}")
            if found:
                # Some sentence may have been padded on an url. Remove padded url.
                sentence = re.sub(r"<|href\S+", "", sentence)
                yield sentence.strip(), entities

    def __iter__(self):
        """Yield cleaned document."""
        yield from self.clean_entities()

    def fb15k237(
        self,
        folder,
        size,
        tokenizer,
        single_token_entities=False,
        mapping_single_token_entities={},
    ):
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

                for i, _ in enumerate(sentence.split("|")):

                    if i % 2:

                        if e_pos == e:

                            entity = entities[e]

                            new_sentence = sentence.split("|")

                            # Single token entities
                            if not single_token_entities or not mapping_single_token_entities:
                                new_sentence[i] = f"| {entity} |"
                            else:
                                new_sentence[
                                    i
                                ] = f"| {mapping_single_token_entities.get(entity, entity)} |"

                            # Remove unicodes caracters:
                            new_sentence = (
                                "".join(new_sentence).strip().replace("  ", " ").replace("\\", "")
                            )
                            new_sentence = new_sentence.encode("ascii", "ignore").decode()

                            # IF || * 2 in sentence truncated to 500 caracters:
                            tokenized_sentence = tokenizer.encode(
                                new_sentence, truncation=True, max_length=500
                            )

                            if tokenized_sentence.count(self.id_sep) == 2:
                                # new_sentence = (".".join(tokenizer.decode(tokenized_sentence).split(".")[:-1]) + ".").replace("[CLS] ", "")
                                found = True
                                entities_found[entity] += 1

                        e_pos += 1

                if found:
                    break

            if found:

                if single_token_entities:
                    entity = mapping_single_token_entities.get(entity, entity)

                n_iter += 1
                batch.append({"sentence": new_sentence, "entity": entity})

            if n_iter == size:
                # Export processed sentences.
                with open(os.path.join(folder, f"{filename}.json"), "w") as fp:
                    json.dump(batch, fp, indent=4)
                filename += 1
                n_iter = 0
                batch = []

        with open(f"{folder}_entities.json", "w") as fp:
            json.dump(entities_found, fp, indent=4)

    def stats(self, tokenizer):

        stats = collections.defaultdict(int)

        for sentence, entities in tqdm.tqdm(self, position=0, desc="Exporting sentences."):

            # Select random entities as candidates
            entities_order = [_ for _ in range(len(entities))]
            random.shuffle(entities_order)
            # Check if there at least one entities that match the condition:
            # Single token mention.
            for e in entities_order:
                e_pos = 0

                for i, mention in enumerate(sentence.split("|")):
                    if i % 2:
                        if e_pos == e:
                            stats[len(tokenizer.tokenize(mention))] += 1
                        e_pos += 1

        return stats
