#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import stanza
from collections import namedtuple


# In[17]:


HEAD = 'head'
POS = 'pos'
Info = namedtuple('Info', [HEAD, POS])
WordNerInfo = namedtuple('WordNerInfo', ['text', 'ner_definition', 'root'])
ROOT = -1
SINGLETON = 'S-'
BEGIN = 'B-'
INSIDE = 'I-'
OUTSIDE = 'O'
END = 'E-'
ner_translator = {
    'GPE': 'מדינה',
    'PER': 'שם',
    'ORG': 'ארגון',
    'LOC': 'מיקום',
    'DUC': 'מוצר',
    'EVE': 'אירוע',
    'ANG': 'שפה',
    'FAC': 'מתקן',
    'WOA': 'יצירת אומנות',
    'OCC': 'מקצוע',
    'PRI': 'פרס',
}


# In[18]:


class Processor:
    def __init__(self):
        self.heb_nlp = stanza.Pipeline(lang='he', processors='tokenize,mwt,pos,lemma,depparse')
        #replace MY_TOKEN with the token you got from the langndata website
        self.yap_token="21e79c010599d991fd815b93048b245b"
    
    def get_stanza_analysis(self, text):
        text += " XX"
        doc=self.heb_nlp(text)
        lst=[]
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features=[(word.text,
                               word.lemma,
                               word.upos,
                               word.xpos,
                               word.head,
                               word.deprel,
                               word.feats)]

                    df=pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel","feats"])
                    lst.append(df)
        tot_df=pd.concat(lst, ignore_index=True)
        tot_df=tot_df.shift(1).iloc[1:]
        tot_df["head"]=tot_df["head"].astype(int)
#         print(tot_df.head(50))
        return tot_df['text'], tot_df['head'], tot_df['upos']
        


# In[19]:


def get_verb_name_for_verb(verb):
    verb_names_db = {
        'נולד': 'לידה',
        'גר': 'מגורים',
        'יסד': 'יסוד',
        'הקים': 'הקמה',
        'עבד': 'עבודה',
        'זכה': 'זכיה',
    }
    return verb_names_db[verb]


# In[20]:


# stanza.download('he')
# stanza_nlp = stanza.Pipeline('he')


# In[21]:


text="""
אלברט איינשטיין נולד בגרמניה וגר בשוויץ
"""
processor=Processor()
processor.get_stanza_analysis(text)


# In[22]:


class SemanticTree:
    def __init__(self, text):
        self.text = text
        self.processor = Processor()
        
    def parse_text(self):
        parsed_text, tree, pos = self.processor.get_stanza_analysis(self.text)
        word_list = list(zip(list(parsed_text), map(lambda head: head - 1, list(tree)), list(pos)))
        self.tree = {word: Info(head, pos) for word, head, pos in word_list}
        self.parsed_text = parsed_text
        print('PARSED_TEXT', self.parsed_text, type(self.parsed_text))
        
    def __str__(self):
        tree_rep = '{\n'
        for word, info in self.tree.items():
            tree_rep += '{}: {}\n'.format(word, info)
        tree_rep += '}\n'
        return tree_rep
        
    def is_verb(self, word):
        return self.tree[word].pos == 'VERB'
    
    def is_root(self, word):
        return self.tree[word].head == ROOT
    
    def get_word_in_index(self, index):
        return list(self.tree.keys())[index]
        
    def find_verb_root(self, word):
        while(not self.is_root(word)):
            if self.is_verb(word):
                return word
            word = self.get_word_in_index(self.tree[word].head)
        return word
    
    def build_ner_for_text(self, ner):
        self.ner = ner
        
    def cluster_text_by_ner(self):
        text_with_ner = list(zip(self.parsed_text, self.ner))
        self.clustered_text = []
        for text, ner in text_with_ner:
            ner = ner.split('^')[-1]
            if ner.startswith(SINGLETON) or ner.startswith(BEGIN):
                self.clustered_text.append((text, ner[2:]))
            elif ner.startswith(OUTSIDE):
                self.clustered_text.append((text, OUTSIDE))
            else:  # INSIDE or END
                previous_text, previous_ner = self.clustered_text.pop()
                united_text = '{} {}'.format(previous_text, text)
                self.clustered_text.append((united_text, previous_ner))
                
                
    def _get_info_for_word_cluster(self, word_cluster):
        text, ner = word_cluster
        root = self.find_verb_root(text.split(' ')[0])
#         if self.is_verb(root):
#             root = get_verb_name_for_verb(root)
        ner_definition = ner_translator[ner]
        return WordNerInfo(text, ner_definition, root)
    
    def _is_word_interesting(self, ner):
        return ner != OUTSIDE
        
    def get_interesting_words_info(self):
        """
        In the future - think what happens when there is more than one name (אלברט דיבר עם ניקו).
        We need to take into account also the POS of the word.
        """
        interesting_words = list(filter(lambda ner_word: 
                                        self._is_word_interesting(ner_word[1]), self.clustered_text))
        interesting_roots = [self._get_info_for_word_cluster(word) for word in interesting_words]
        return interesting_roots
    
    def build_info_dict(self, interesting_words_info):
        return {
            '{}{}'.format(word_info.ner_definition, 
                             ' - ' + word_info.root if word_info.ner_definition != 'שם' else ''): word_info.text
            for word_info in interesting_words_info
        }
    
    def add_date_tags(self):
        pass


# In[23]:


def str_interesting_words(words):
    repr_str = '[\n'
    for word in words:
        repr_str += 'טקסט: {}, NER: {}, שורש: {}\n'.format(word.text, word.ner_definition, word.root)
    repr_str += ']\n'
    return repr_str


# In[26]:


text="""
אלברט איינשטיין נולד בגרמניה וגר בשוויץ
"""
text = 'מאי נולדה ב 21 ב אפריל 1984 ל פנה"ס ו התחילה ללמוד ב טכניון ב כ״ט ב שנת ה׳תשע"ה'
tree = SemanticTree(text)
tree.parse_text()
print(tree)
# word = 'שוויץ'
# print('{} -> {}'.format(word, tree.find_verb_root(word)))
# word = 'גרמניה'
# print('{} -> {}'.format(word, tree.find_verb_root(word)))
tree.build_ner_for_text(['B-PER', 'E-PER', 'O', 'O', 'S-GPE', 'O^O', 'O', 'O', 'S-GPE'])
tree.add_date_tags()
tree.cluster_text_by_ner()
print(tree.clustered_text)
interesting_words_info = tree.get_interesting_words_info()
print(str_interesting_words(interesting_words_info))
print(tree.build_info_dict(interesting_words_info))


# In[ ]:


ner_results = ['B-PER', 'E-PER', 'O', 'O', 'S-GPE', 'O^O', 'O', 'O', 'S-GPE']


# In[245]:


text="""
אברהם כהן עבד בחברת אגד כנהג אוטובוס בישראל
"""
tree = SemanticTree(text)
tree.parse_text()
# print(tree.tree)
# word = 'שוויץ'
# print('{} -> {}'.format(word, tree.find_verb_root(word)))
# word = 'גרמניה'
# print('{} -> {}'.format(word, tree.find_verb_root(word)))
tree.build_ner_for_text(['B-PER', 'E-PER', 'O', 'O', 'O', 'B-ORG', 'O^O', 'O^O', 'O', 'O^S-GPE'])
tree.cluster_text_by_ner()
print(tree.clustered_text)
interesting_words_info = tree.get_interesting_words_info()
for title, word in tree.build_info_dict(interesting_words_info).items():
    print(title + ': ' + word)


# In[255]:


text="""
אלברט איינשטיין זכה ב פרס נובל
"""
tree = SemanticTree(text)
tree.parse_text()
# print(tree.tree)
# word = 'שוויץ'
# print('{} -> {}'.format(word, tree.find_verb_root(word)))
# word = 'גרמניה'
# print('{} -> {}'.format(word, tree.find_verb_root(word)))
tree.build_ner_for_text(['B-PER', 'E-PER', 'O', 'O', 'B-PRI', 'E-PRI'])
tree.cluster_text_by_ner()
print(tree.clustered_text)
interesting_words_info = tree.get_interesting_words_info()
for title, word in tree.build_info_dict(interesting_words_info).items():
    print(title + ': ' + word)

