import re as regex
import string

# ב-1997 ב1997 התשע״ה תשע״ה
DATE_TAG = 'DATE'
date_related_words = ['ינואר', 'פברואר', 'מרץ', 'מרס', 'מארס', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט',
                      'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר', 'שנה', 'שנת', 'תשרי', 'חשוון', 'חשון', 'כסלו',
                      'טבת', 'שבט', 'אדר', 'ניסן', 'אייר', 'סיוון', 'סיון', 'תמוז', 'אב',
                      'אלול']
year_regex = r'([0-9]{4}\b|\u05D4?[׳\']?[\u05E7-\u05EA]{2}[\u05D9-\u05E6][״"][\u05D0-\u05D8]\b)'
hebrew_day_regex = r'([\u05D0-\u05D8][\'׳]|[\u05D9-\u05E1]["״][\u05D0-\u05D8]\b)'


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


# def get_indices_to_tag(index, text):
#     splitted_text = text.split()
#     epsilon_space_indices = [index + gap for gap in range(-2, 3) if 0 <= index + gap < len(splitted_text)]
#     print(epsilon_space_indices)
#     number_indices = {ind for ind in epsilon_space_indices if does_contain_digits(splitted_text[ind])}
#     if number_indices:
#         number_indices.add(index)
#         all_indices_range = sorted(number_indices)
#         print(number_indices)
#         return range(all_indices_range[0], all_indices_range[-1] + 1)
#     return []

def is_date_related(word):
    return len(word) == 1 \
           or word in date_related_words \
           or does_contain_digits(word) \
           or regex.match(year_regex, word) \
           or regex.match(hebrew_day_regex, word)


def get_date_related_indices(first_index_to_check, last_index, split_text):
    date_related_range = []
    for index in range(last_index, first_index_to_check - 1, -1):
        print(index, split_text[index], is_date_related(split_text[index]))
        if is_date_related(split_text[index]):
            date_related_range.append(index)
        else:
            break
    return sorted(date_related_range)


def get_indices_to_tag(index, text):
    split_text = text.split()
    first_index_to_check = max(index - 5, 0)
    return get_date_related_indices(first_index_to_check, index, split_text)


# text = '25.08.2020 is the birth date of Saifun'
text = '2012 מאי נולדה ב 25 ב אוגוסט 1997 ל פנה"ס ו התחילה ללמוד ב טכניון ב כ״ט ב שנת ה׳תשע"ה'
ner = ['O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-ORG', 'O', 'O', 'O', 'O', 'O']
span_to_index = build_span_to_index_dict(text)
print(span_to_index)
all_month_indices = []
# date_numeric_regex = regex.compile(r'\d{1,4}([.\-/])\d{1,2}([.\-/])\d{1,4}')
# a = date_numeric_regex.match(text)
# print(type(a.group()))
# for month in date_words_regular_behavior:
#     month_regex = regex.compile(r'\w*{}\w*'.format(month))
#     print(month_regex)
#     matched_text = month_regex.finditer(text)
#     all_month_indices.extend([span_to_index[result.span()] for result in matched_text])
# print(all_month_indices)
# for month_index in all_month_indices:
#     indices_to_tag_as_date = get_indices_to_tag(month_index, text)
#     if all([(ner[index].startswith('O') or ner[index] == DATE_TAG) for index in indices_to_tag_as_date]):
#         for index in indices_to_tag_as_date:
#             ner[index] = DATE_TAG


year_regex_compiled = regex.compile(year_regex)
matched_year = year_regex_compiled.finditer(text)
all_year_indices = [span_to_index[result.span()] for result in matched_year]
print(all_year_indices)
for year_index in all_year_indices:
    indices_to_tag_as_date = get_indices_to_tag(year_index, text)
    print(indices_to_tag_as_date)
    if all([(ner[index].startswith('O') or ner[index] == DATE_TAG) for index in indices_to_tag_as_date]):
        for index in indices_to_tag_as_date:
            ner[index] = DATE_TAG
print(ner)

# hebrew_year_regex = r'\u05D4?[׳\']?[\u05E7-\u05EA]{2}[\u05D9-\u05E6][״"][\u05D0-\u05D8]\b'
# year_regex_compiled = regex.compile(hebrew_year_regex)
# matched_text = year_regex_compiled.finditer(text)
# all_year_indices = [span_to_index[result.span()] for result in matched_text]
# print(all_year_indices)
# for year_index in all_year_indices:
#     indices_to_tag_as_date = get_indices_to_tag(year_index, text)
#     print(indices_to_tag_as_date)
#     if all([(ner[index].startswith('O') or ner[index] == DATE_TAG) for index in indices_to_tag_as_date]):
#         for index in indices_to_tag_as_date:
#             ner[index] = DATE_TAG
print(ner)
