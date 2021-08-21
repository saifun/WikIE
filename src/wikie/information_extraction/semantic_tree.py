from .stanza_processor import Processor
from .date_recognition import enrich_ner_tags_with_dates
from .text_ner_tagging import get_ner_for_text
from .consts import Info, ROOT, SINGLETON, BEGIN, OUTSIDE, WordNerInfo, ner_translator


class SemanticTree:
    def __init__(self, text, ner_model):
        self.text = text
        self.ner_model = ner_model
        self.processor = Processor()

    def get_extracted_information_for_text(self):
        self.parse_text()
        self.build_ner_for_text()
        self.cluster_text_by_ner()
        interesting_words_info = self.get_interesting_words_info()
        return self.build_info_dict(interesting_words_info)

    def parse_text(self):
        parsed_text, tree, pos = self.processor.get_stanza_analysis(self.text)
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos)))
        self.tree = {word: Info(head, pos) for word, head, pos in word_list}
        self.parsed_text = parsed_text
        print('PARSED_TEXT', self.parsed_text, type(self.parsed_text))

    def __str__(self):
        tree_rep = '{\n'
        for word, info in self.tree.items():
            tree_rep += '{}: {}\n'.format(word, info)
        tree_rep += '}\n'
        return tree_rep

    def is_verb(self, word):
        return self.tree[word].pos == 'VERB'

    def is_root(self, word):
        return self.tree[word].head == ROOT

    def get_word_in_index(self, index):
        return list(self.tree.keys())[index]

    def find_verb_root(self, word):
        while (not self.is_root(word)):
            if self.is_verb(word):
                return word
            word = self.get_word_in_index(self.tree[word].head)
        return word

    def build_ner_for_text(self):
        self.ner = get_ner_for_text(self.text, self.ner_model)

    def cluster_text_by_ner(self):
        text_with_ner = list(zip(self.parsed_text, self.ner))
        self.clustered_text = []
        for text, ner in text_with_ner:
            ner = ner.split('^')[-1]
            if ner.startswith(SINGLETON) or ner.startswith(BEGIN):
                self.clustered_text.append((text, ner[2:]))
            elif ner.startswith(OUTSIDE):
                self.clustered_text.append((text, OUTSIDE))
            else:  # INSIDE or END
                previous_text, previous_ner = self.clustered_text.pop()
                united_text = '{} {}'.format(previous_text, text)
                self.clustered_text.append((united_text, previous_ner))

    def _get_info_for_word_cluster(self, word_cluster):
        text, ner = word_cluster
        root = self.find_verb_root(text.split(' ')[0])
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
                                        self._is_word_interesting(ner_word[1]), self.clustered_text))
        interesting_roots = [self._get_info_for_word_cluster(word) for word in interesting_words]
        return interesting_roots

    def build_info_dict(self, interesting_words_info):
        return {
            '{}{}'.format(word_info.ner_definition,
                          ' - ' + word_info.root if word_info.ner_definition != 'שם' else ''): word_info.text
            for word_info in interesting_words_info
        }

    def add_date_tags(self):
        self.ner = enrich_ner_tags_with_dates(self.text, self.ner)
