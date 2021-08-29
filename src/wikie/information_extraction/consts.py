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

punctuation = string.punctuation + chr(8211)

ner_translation = {
    'GPE': 'מקום',
    'PER': 'שם',
    'ORG': 'ארגון',
    'LOC': 'מיקום',
    'DUC': 'מוצר',
    'EVE': 'אירוע',
    'ANG': 'שפה',
    'FAC': 'מתקן',
    'WOA': 'יצירת אומנות',
    'OCC': 'מקצוע',
    'DATE': 'תאריך'
}


def ner_translator(ner):
    if ner.startswith(DATE_TAG):
        return 'תאריך'
    return ner_translation.get(ner)


"""
Date extraction related consts
"""
DATE_TAG = 'DATE'
DATE_TEXT_MAX_GAP = 5
date_related_words = ['ינואר', 'פברואר', 'מרץ', 'מרס', 'מארס', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט',
                      'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר', 'שנה', 'שנת', 'תשרי', 'חשוון', 'חשון', 'כסלו',
                      'טבת', 'שבט', 'אדר', 'ניסן', 'אייר', 'סיוון', 'סיון', 'תמוז', 'אב',
                      'אלול']
year_regex = r'([0-9]{4}\b|\u05D4?[׳\']?[\u05E7-\u05EA]{2}[\u05D9-\u05E6][״"][\u05D0-\u05D8]\b)'
hebrew_day_regex = r'([\u05D0-\u05D8][\'׳]|[\u05D9-\u05E1]["״][\u05D0-\u05D8]\b)'
