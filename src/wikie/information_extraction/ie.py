from .semantic_tree import SemanticTree
from .text_ner_tagging import NERModel


class IE:
    def __init__(self):
        self.ner_model = NERModel()
        self.ner_model.load_labels()

    def extract_text_information(self, text):
        semantic_tree = SemanticTree(text, self.ner_model)
        interesting_words_info = semantic_tree.get_extracted_information_for_text()
        for title, word in interesting_words_info.items():
            print(title + ': ' + word)
        return interesting_words_info
