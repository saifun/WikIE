from semantic_tree import SemanticTree

def extract_text_information(text):
    semantic_tree = SemanticTree(text)
    semantic_tree.parse_text()
    # TODO: replace with call to NER-BERT
    semantic_tree.build_ner_for_text(['B-PER', 'E-PER', 'O', 'O', 'O', 'B-ORG', 'O^O', 'O^O', 'O', 'O^S-GPE'])
    semantic_tree.cluster_text_by_ner()
    interesting_words_info = semantic_tree.get_interesting_words_info()
    for title, word in semantic_tree.build_info_dict(interesting_words_info).items():
        print(title + ': ' + word)
    return interesting_words_info