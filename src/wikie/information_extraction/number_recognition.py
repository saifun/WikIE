from .date_recognition import does_contain_digits
from .consts import OUTSIDE, NUMBER_TAG

def enrich_ner_tags_with_numbers(parsed_text, ner):
    text = parsed_text.tolist()
    print(text)
    for index, word in enumerate(text):
        if does_contain_digits(word) and ner[index].startswith(OUTSIDE):
            ner[index] = NUMBER_TAG
    return ner