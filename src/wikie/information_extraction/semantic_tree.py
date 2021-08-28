from collections import defaultdict

from .stanza_processor import Processor
from .date_recognition import enrich_ner_tags_with_dates
from .text_ner_tagging import get_ner_for_text
from .consts import Info, ROOT, OUTSIDE, WordNerInfo, ner_translator, NER


class SemanticTree:
    def __init__(self, text, ner_model):
        self.text = text
        self.ner_model = ner_model
        self.processor = Processor()

    def get_extracted_information_for_text(self):
        self.parse_text()
        self.build_ner_for_text()
        self.add_date_tags()
        self.cluster_text_by_ner()
        interesting_words_info = self.get_interesting_words_info()
        organized_info = self.build_info_representation(interesting_words_info)
        return organized_info

    def parse_text(self):
        parsed_text, tree, pos = self.processor.get_stanza_analysis(self.text)
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos)))
        self.tree = {index: Info(word, head, pos) for index, (word, head, pos) in enumerate(word_list)}
        self.parsed_text = parsed_text

    def __str__(self):
        tree_rep = '{\n'
        for index, info in self.tree.items():
            tree_rep += '{}: {}\n'.format(index, info)
        tree_rep += '}\n'
        return tree_rep

    def is_verb(self, word_index):
        return self.tree[word_index].pos == 'VERB'

    def is_root(self, word_index):
        return self.tree[word_index].head == ROOT

    def get_word_in_index(self, index):
        return self.tree[index].word

    def find_verb_root(self, word_index):
        while (not self.is_root(word_index)):
            if self.is_verb(word_index):
                return word_index
            word_index = self.tree[word_index].head
        return word_index

    def build_ner_for_text(self):
        self.ner = get_ner_for_text(self.parsed_text, self.ner_model)

    @staticmethod
    def _get_bare_ner(ner):
        """
        Peels all ner prefixes and returns the core ner.
        """
        return ner.split('^')[-1].split('-')[-1]

    def cluster_text_by_ner(self):
        text_with_ner = list(zip(self.parsed_text, self.ner))
        self.clustered_text = []
        for index, (text, ner) in enumerate(text_with_ner):
            ner = self._get_bare_ner(ner)
            if self.clustered_text and self.clustered_text[-1][NER] == ner:
                previous_text, previous_ner, previous_index = self.clustered_text.pop()
                united_text = '{} {}'.format(previous_text, text)
                self.clustered_text.append((united_text, previous_ner, previous_index))
            else:
                self.clustered_text.append((text, ner, index))

    def _get_info_for_word_cluster(self, word_cluster):
        text, ner, index = word_cluster
        root = self.get_word_in_index(self.find_verb_root(index))
        ner_definition = ner_translator[ner]
        return WordNerInfo(text, ner_definition, root)

    def _is_word_interesting(self, ner):
        return ner != OUTSIDE

    def get_interesting_words_info(self):
        """
        In the future - think what happens when there is more than one name (אלברט דיבר עם ניקו).
        We need to take into account also the POS of the word.
        """
        interesting_words = list(filter(lambda ner_word:
                                        self._is_word_interesting(ner_word[NER]), self.clustered_text))
        interesting_roots = [self._get_info_for_word_cluster(word) for word in interesting_words]
        return interesting_roots

    def build_info_representation(self, interesting_words_info):
        unusual_ner_tagging = ['שם', 'מקצוע']
        info = defaultdict(set)
        for word_info in interesting_words_info:
            info_to_present = word_info.text if word_info.ner_definition in unusual_ner_tagging else (
                word_info.text, word_info.root)
            info[word_info.ner_definition].add(info_to_present)
        return dict(info)

    def add_date_tags(self):
        self.ner = enrich_ner_tags_with_dates(self.parsed_text, self.ner)
