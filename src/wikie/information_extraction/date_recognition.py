import string
import re as regex
from .consts import DATE_TAG, date_related_words, year_regex, hebrew_day_regex, DATE_TEXT_MAX_GAP, punctuation


def build_span_to_index_dict(text):
    current_pos = 0
    span_to_index = {}
    for index, word in enumerate(text.split()):
        end_pos = current_pos + len(word)
        span_to_index[(current_pos, end_pos)] = index
        current_pos = end_pos + 1
    return span_to_index


def does_contain_digits(word):
    return any([digit in word for digit in string.digits])


def is_date_related(word):
    return word not in punctuation \
           and (len(word) == 1
                or word in date_related_words
                or word[1:] in date_related_words
                or does_contain_digits(word)
                or regex.match(year_regex, word)
                or regex.match(hebrew_day_regex, word))


def get_date_related_indices(first_index_to_check, last_index, split_text):
    date_related_range = []
    for index in range(last_index, first_index_to_check - 1, -1):
        if is_date_related(split_text[index]):
            date_related_range.append(index)
        else:
            break
    return sorted(date_related_range)


def get_indices_to_tag(index, text, prev_year_index):
    split_text = text.split()
    first_index_to_check = max(index - DATE_TEXT_MAX_GAP, 0, prev_year_index + 1)
    return get_date_related_indices(first_index_to_check, index, split_text)


def get_year_indices(text):
    # TODO: what if the text contains בתשע״ד and stanza didn't catch it?
    span_to_index = build_span_to_index_dict(text)
    year_regex_compiled = regex.compile(year_regex)
    matched_year = year_regex_compiled.finditer(text)
    return [span_to_index[result.span()] for result in matched_year]


def enrich_ner_tags_with_dates(parsed_text, ner):
    text = ' '.join(parsed_text.tolist())
    all_year_indices = get_year_indices(text)
    prev_year_index = -1
    date_pattern_index = 0
    for year_index in all_year_indices:
        indices_to_tag_as_date = get_indices_to_tag(year_index, text, prev_year_index)
        if all([(ner[index].startswith('O') or ner[index].startswith(DATE_TAG)) for index in indices_to_tag_as_date]):
            for index in indices_to_tag_as_date:
                ner[index] = '{}_{}'.format(DATE_TAG, date_pattern_index)
        prev_year_index = year_index
        date_pattern_index += 1
    return ner
