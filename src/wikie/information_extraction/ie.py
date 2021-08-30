from collections import defaultdict
from functools import reduce

from .semantic_tree import SemanticTree
from .text_ner_tagging import NERModel


class IE:
    def __init__(self):
        self.ner_model = NERModel()
        self.ner_model.load_labels()

    def merge_dicts(self, dict_a, dict_b):
        if dict_a is None:
            dict_a = {}
        default_first_dict = defaultdict(set, dict_a)
        for key, value in dict_b.items():
            default_first_dict[key].update(value)
        return default_first_dict

    def merge_multiple_dicts(self, dicts_list):
        return dict(reduce(self.merge_dicts, dicts_list))

    def extract_text_information(self, text):
        sentences = list(filter(lambda sentence: sentence.strip(), text.split('.')))
        semantic_trees = [SemanticTree(sentence, self.ner_model) for sentence in sentences]
        interesting_words_info = self.merge_multiple_dicts(
            [semantic_tree.get_extracted_information_for_text() for semantic_tree in semantic_trees])
        return interesting_words_info

    def extract_information_from_file(self, file_path):
        with open(file_path, 'r') as input_file:
            text = input_file.read()
            return self.extract_text_information(text)
