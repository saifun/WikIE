from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification, Trainer
tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
import numpy as np

import torch

class HebrewNERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

trainer = BertForSequenceClassification.from_pretrained('./alephbert_ner_occ_morph')

trainer = Trainer(trainer)

labels = []
with open('labels.txt', 'r') as file:
    for line in file:
        labels.append(line.strip())

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)

test_s = "יוסי שטיינמץ הוא נגן קלרינט"
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
label_encoder.inverse_transform(test_s_pred)