import string
from collections import namedtuple

"""
Semantic representation related consts
"""
HEAD = 'head'
POS = 'pos'
WORD = 'word'
Info = namedtuple('Info', [WORD, HEAD, POS])
WordNerInfo = namedtuple('WordNerInfo', ['text', 'ner_definition', 'root'])
ROOT = -1
OUTSIDE = 'O'
END = 'E-'
NER = 1
HEBREW_DASH = chr(8211)
punctuation = string.punctuation + HEBREW_DASH

"""
Date extraction related consts
"""
DATE_TAG = 'DATE'
BIRTH_DATE_TAG = DATE_TAG + '_BIRTH'
DEATH_DATE_TAG = DATE_TAG + '_DEATH'

DATE_TEXT_MAX_GAP = 5
date_related_words = ['ינואר', 'פברואר', 'מרץ', 'מרס', 'מארס', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט',
                      'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר', 'שנה', 'שנת', 'תשרי', 'חשוון', 'חשון', 'כסלו',
                      'טבת', 'שבט', 'אדר', 'ניסן', 'אייר', 'סיוון', 'סיון', 'תמוז', 'אב',
                      'אלול', 'פנה"ס', 'לפנה"ס']
BCE_regex = r'[\u05E4][\u05E0][\u05D4][״"][\u05E1]\b'
numeric_year_regex = r'[0-9]{4}\b'
hebrew_decade_year_regex = r'\u05D4?[׳\'\']?[\u05E7-\u05EA]{2}[״"][\u05D9-\u05E6]\b'
hebrew_year_regex = r'\u05D4?[׳\'\']?[\u05E7-\u05EA]{2}[\u05D9-\u05E6]?[״"][\u05D0-\u05D8]\b'
year_regex = f'({numeric_year_regex}|{hebrew_year_regex}|{hebrew_decade_year_regex}|{BCE_regex})'
hebrew_day_regex = r'([\u05D0-\u05D8][\'\'׳]|[\u05D9-\u05E1]["״][\u05D0-\u05D8]\b)'

"""
Tags related consts
"""

NAME = 'שם'
OCCUPATION = 'מקצוע'
ner_translation = {
    'GPE': 'מקום',
    'PER': NAME,
    'ORG': 'ארגון',
    'LOC': 'מיקום',
    'DUC': 'מוצר',
    'EVE': 'אירוע',
    'ANG': 'שפה',
    'FAC': 'מתקן',
    'WOA': 'יצירת אומנות',
    'OCC': OCCUPATION,
    'DATE': 'תאריך'
}


def ner_translator(ner):
    if ner.startswith(BIRTH_DATE_TAG):
        return 'תאריך לידה'
    if ner.startswith(DEATH_DATE_TAG):
        return 'תאריך פטירה'
    if ner.startswith(DATE_TAG):
        return 'תאריך'
    return ner_translation.get(ner)


UNUSUAL_TAGS = [NAME, OCCUPATION, ner_translator(BIRTH_DATE_TAG), ner_translator(DEATH_DATE_TAG)]


def is_unusual_tag(ner_tag):
    return ner_tag in UNUSUAL_TAGS
