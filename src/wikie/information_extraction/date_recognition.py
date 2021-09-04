import string
import re as regex
from .consts import DATE_TAG, date_related_words, year_regex, hebrew_day_regex, DATE_TEXT_MAX_GAP, punctuation, \
    HEBREW_DASH, BIRTH_DATE_TAG, DEATH_DATE_TAG


def build_index_conversion_dicts(text):
    current_pos = 0
    char_index_to_word_index = {}
    word_index_to_word_start = {}
    for index, word in enumerate(text.split()):
        end_pos = current_pos + len(word)
        next_word_start = end_pos + 1
        for sub_index in range(current_pos, next_word_start):
            char_index_to_word_index[sub_index] = index
        word_index_to_word_start[index] = current_pos
        current_pos = next_word_start
    return char_index_to_word_index, word_index_to_word_start


def does_contain_digits(word):
    return any([digit in word for digit in string.digits])


def is_date_related(word):
    clean_word = word.strip(string.punctuation)
    return clean_word not in punctuation \
           and (len(clean_word) == 1
                or clean_word in date_related_words
                or clean_word[1:] in date_related_words
                or does_contain_digits(clean_word)
                or regex.match(year_regex, clean_word)
                or regex.match(hebrew_day_regex, clean_word)
                or regex.match(hebrew_day_regex, clean_word[1:]))


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


def get_year_indices(text, char_index_to_word_index):
    year_regex_compiled = regex.compile(year_regex)
    matched_year = year_regex_compiled.finditer(text)
    all_year_indices = [(char_index_to_word_index[result.start()], result.start()) for result in matched_year]
    return all_year_indices


def get_all_occurrences(character, text):
    return list(map(lambda span: span.start(), regex.finditer(character, text)))


def get_date_related_punctuation(text):
    left_parentheses_indices = get_all_occurrences(r'\(', text)
    right_parentheses_indices = get_all_occurrences(r'\)', text)
    dash_indices = get_all_occurrences(HEBREW_DASH, text) + get_all_occurrences('-', text)
    return left_parentheses_indices, right_parentheses_indices, dash_indices


def get_closest_punctuation_on_side(year_index, punctuation_indices, before=True):
    relevant_indices = filter_words_by_position(year_index, punctuation_indices, before=before)
    if relevant_indices:
        if before:
            return relevant_indices[-1]
        else:
            return relevant_indices[0]
    return None


def filter_words_by_position(year_index, words_indices, before=True):
    return list(
        filter(lambda word_index: word_index < year_index if before else word_index > year_index, words_indices))


def is_death_year(year_index, opening_parentheses_before, closing_parentheses_after, dash_before, char_year_indices):
    year_indices_before = filter_words_by_position(year_index, char_year_indices)
    return (opening_parentheses_before
            and closing_parentheses_after
            and dash_before
            and opening_parentheses_before < dash_before < year_index < closing_parentheses_after
            and any([opening_parentheses_before < year_before < dash_before for year_before in year_indices_before]))


def get_date_type(year_index, left_parentheses_indices, right_parentheses_indices, dash_indices, all_year_indices):
    char_year_indices = [year_tuple[1] for year_tuple in all_year_indices]
    opening_parentheses_before = get_closest_punctuation_on_side(year_index, left_parentheses_indices)
    closing_parentheses_after = get_closest_punctuation_on_side(year_index, right_parentheses_indices, before=False)
    dash_before = get_closest_punctuation_on_side(year_index, dash_indices)
    if is_death_year(year_index, opening_parentheses_before, closing_parentheses_after, dash_before, char_year_indices):
        return DEATH_DATE_TAG
    if opening_parentheses_before and closing_parentheses_after:
        return BIRTH_DATE_TAG
    return DATE_TAG


def enrich_ner_tags_with_dates(parsed_text, ner):
    text = ' '.join(parsed_text.tolist())
    char_index_to_word_index, word_index_to_word_start = build_index_conversion_dicts(text)
    all_year_indices = get_year_indices(text, char_index_to_word_index)
    prev_year_index = -1
    date_pattern_index = 0
    left_parentheses_indices, right_parentheses_indices, dash_indices = get_date_related_punctuation(text)
    for year_index, char_year_index in all_year_indices:
        indices_to_tag_as_date = get_indices_to_tag(year_index, text, prev_year_index)
        if any([ner[index].startswith('O') for index in indices_to_tag_as_date]):
            for index in indices_to_tag_as_date:
                date_type_tag = get_date_type(word_index_to_word_start[indices_to_tag_as_date[0]],
                                              left_parentheses_indices,
                                              right_parentheses_indices,
                                              dash_indices,
                                              all_year_indices)
                ner[index] = '{}_{}'.format(date_type_tag, date_pattern_index)
        prev_year_index = year_index
        date_pattern_index += 1
    return ner
