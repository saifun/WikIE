import torch
import numpy as np
from sklearn import preprocessing
from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification, Trainer


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


class NERModel:
    def __init__(self):
        trainer = BertForSequenceClassification.from_pretrained('wikie/model_data/alephbert_ner_occ_morph')
        self.trained_model = Trainer(trainer)
        self.tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')

    def load_labels(self):
        with open('wikie/information_extraction/labels.txt', 'r') as labels_file:
            self.labels = list(map(lambda label: label.strip(), labels_file.readlines()))
            self.label_encoder = preprocessing.LabelEncoder()
            self.label_encoder.fit(self.labels)

    def predict(self, dataset):
        text_prediction_probabilities, _, _ = self.trained_model.predict(dataset)
        text_predictions = np.argmax(text_prediction_probabilities, axis=1)
        return self.label_encoder.inverse_transform(text_predictions)


def get_ner_for_text(text, ner_model):
    tokenized_text = ner_model.tokenizer(text.split(), truncation=True, padding=True)
    text_dataset = HebrewNERDataset(tokenized_text, [0 for _ in range(len(tokenized_text))])
    return ner_model.predict(text_dataset)
