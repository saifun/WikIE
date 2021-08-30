import string
import re as regex
from .consts import DATE_TAG, date_related_words, year_regex, hebrew_day_regex, DATE_TEXT_MAX_GAP, punctuation, \
    HEBREW_DASH, BIRTH_DATE_TAG, DEATH_DATE_TAG


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
    all_year_indices = [(span_to_index[result.span()], result.start()) for result in matched_year]
    return all_year_indices


def get_all_occurrences(character, text):
    return list(map(lambda span: span.start(), regex.finditer(character, text)))


def get_date_related_punctuation(text):
    left_parentheses_indices = get_all_occurrences(r'\(', text)
    right_parentheses_indices = get_all_occurrences(r'\)', text)
    dash_indices = get_all_occurrences(HEBREW_DASH, text) + get_all_occurrences('-', text)
    return left_parentheses_indices, right_parentheses_indices, dash_indices


def does_punctuation_exist_within_side(year_index, punctuation_indices, before=True):
    return any(
        filter(lambda punctuation_index: punctuation_index < year_index if before else punctuation_index > year_index,
               punctuation_indices))


def get_date_type(year_index, left_parentheses_indices, right_parentheses_indices, dash_indices):
    opening_parentheses_before = does_punctuation_exist_within_side(year_index, left_parentheses_indices)
    closing_parentheses_after = does_punctuation_exist_within_side(year_index, right_parentheses_indices, before=False)
    dash_before = does_punctuation_exist_within_side(year_index, dash_indices)
    dash_after = does_punctuation_exist_within_side(year_index, dash_indices, before=False)
    if opening_parentheses_before and closing_parentheses_after and dash_after:
        return BIRTH_DATE_TAG
    if opening_parentheses_before and closing_parentheses_after and dash_before:
        return DEATH_DATE_TAG
    return DATE_TAG


def enrich_ner_tags_with_dates(parsed_text, ner):
    text = ' '.join(parsed_text.tolist())
    all_year_indices = get_year_indices(text)
    prev_year_index = -1
    date_pattern_index = 0
    left_parentheses_indices, right_parentheses_indices, dash_indices = get_date_related_punctuation(text)
    for year_index, char_year_index in all_year_indices:
        indices_to_tag_as_date = get_indices_to_tag(year_index, text, prev_year_index)
        if all([ner[index].startswith('O') for index in indices_to_tag_as_date]):
            for index in indices_to_tag_as_date:
                date_type_tag = get_date_type(char_year_index,
                                              left_parentheses_indices,
                                              right_parentheses_indices,
                                              dash_indices)
                ner[index] = '{}_{}'.format(date_type_tag, date_pattern_index)
        prev_year_index = year_index
        date_pattern_index += 1
    return ner
