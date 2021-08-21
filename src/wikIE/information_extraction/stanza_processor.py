import stanza

class Processor:
    def __init__(self):
        self.heb_nlp = stanza.Pipeline(lang='he', processors='tokenize,mwt,pos,lemma,depparse')
        # replace MY_TOKEN with the token you got from the langndata website
        self.yap_token = "21e79c010599d991fd815b93048b245b"

    def get_stanza_analysis(self, text):
        text += " XX"
        doc = self.heb_nlp(text)
        lst = []
        for sen in doc.sentences:
            for token in sen.tokens:
                for word in token.words:
                    features = [(word.text,
                                 word.lemma,
                                 word.upos,
                                 word.xpos,
                                 word.head,
                                 word.deprel,
                                 word.feats)]

                    df = pd.DataFrame(features, columns=["text", "lemma", "upos", "xpos", "head", "deprel", "feats"])
                    lst.append(df)
        tot_df = pd.concat(lst, ignore_index=True)
        tot_df = tot_df.shift(1).iloc[1:]
        tot_df["head"] = tot_df["head"].astype(int)
        return tot_df['text'], tot_df['head'], tot_df['upos']
