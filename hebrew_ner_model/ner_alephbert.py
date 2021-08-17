#!/bin/python3
# -*- coding: utf-8 -*-
from transformers import BertModel, BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("onlplab/alephbert-base", num_labels=135)
model.save_pretrained("./initial_pretrained")


# %% [code]
dataset = {
    "name": "NEMO Corpus",
    "train_path": "~/code/NEMO-Corpus/data/spmrl/gold/token-multi_gold_train.bmes",
    "dev_path": "~/code/NEMO-Corpus/data/spmrl/gold/token-multi_gold_dev.bmes",
    "test_path": "~/code/NEMO-Corpus/data/spmrl/gold/token-multi_gold_test.bmes",
    'classes': []
}

# %% [code]
labels = []
with open('labels.txt', 'r') as file:
    for line in file:
        labels.append(line.strip())
labels.extend(["OCC", "O^OCC", "O^O^OCC", "O^O^O^OCC"])
print(labels)
dataset['classes'] = labels
#print(len(labels))
#print(labels[61])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-19T11:58:17.782569Z","iopub.execute_input":"2021-07-19T11:58:17.78293Z","iopub.status.idle":"2021-07-19T11:58:18.090415Z","shell.execute_reply.started":"2021-07-19T11:58:17.782883Z","shell.execute_reply":"2021-07-19T11:58:18.089386Z"}}
import pandas as pd


def read_data():
    train = pd.read_csv(dataset['train_path'], sep=' ', engine='python', quoting=3, encoding='utf-8',
                        error_bad_lines=False, names=['word', 'label'])
    dev = pd.read_csv(dataset['dev_path'], sep=' ', engine='python', quoting=3, encoding='utf-8', error_bad_lines=False,
                      names=['word', 'label'])
    test = pd.read_csv(dataset['test_path'], sep=' ', engine='python', quoting=3, encoding='utf-8',
                       error_bad_lines=False, names=['word', 'label'])
    return train, dev, test


train, dev, test = read_data()
# train.to_csv("train_example.csv")

# %% [code]
occupation_file_path = './occupations.txt'
occ_file = open(occupation_file_path, 'r')
occupations_set = set([line.replace('\n', '') for line in occ_file.readlines() if line.replace('\n', '')])

# %% [code]
label_not_ne = ["O", "O^O", "O^O^O", "O^O^O^O"]


# %% [code]
def is_word_an_occupation(word, label):
    if label in label_not_ne:
        num_prefixes = label.count("^", 0, len(label))
        if word[num_prefixes:] in occupations_set:
            return True
    return False


# print(is_word_an_occupation("וכנהג", "O^O^O"))

def replace_occupations_label(dataframe):
    for index, row in dataframe.iterrows():
        if is_word_an_occupation(row["word"], row["label"]):
            row["label"] = row["label"][:-1] + "OCC"


# replace_occupations_label(train)
# testing_df = train[:10]
# replace_occupations_label(testing_df)
# print(testing_df)
# testing_df.loc[testing_df['word'] in ("תהיה","נמצאים"), 'label'] = 'J'
# np.where(testing_df['word'] in ("תהיה","נמצאים"), 'J', testing_df["label"])
# testing_df["label"] = ['J' if word in ("תהיה","נמצאים") else '0' for word in testing_df['word']]
# print(testing_df)
replace_occupations_label(train)
replace_occupations_label(test)
replace_occupations_label(dev)


# for r in label_not_ne:
#     print(r.count("^", 0, len(r)))

def change_label_example():
    train["label"][0] = "O"


# %% [code] {"execution":{"iopub.status.busy":"2021-07-19T11:58:37.06152Z","iopub.execute_input":"2021-07-19T11:58:37.061875Z","iopub.status.idle":"2021-07-19T11:58:37.100641Z","shell.execute_reply.started":"2021-07-19T11:58:37.061843Z","shell.execute_reply":"2021-07-19T11:58:37.099805Z"}}
train[train["label"] == 'O^O^OCC']

# %% [code]
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-19T11:58:46.541693Z","iopub.execute_input":"2021-07-19T11:58:46.542023Z","iopub.status.idle":"2021-07-19T11:58:46.5513Z","shell.execute_reply.started":"2021-07-19T11:58:46.541992Z","shell.execute_reply":"2021-07-19T11:58:46.550284Z"}}
label_encoder.inverse_transform([61])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-19T11:58:48.883781Z","iopub.execute_input":"2021-07-19T11:58:48.884114Z","iopub.status.idle":"2021-07-19T11:58:51.389112Z","shell.execute_reply.started":"2021-07-19T11:58:48.884085Z","shell.execute_reply":"2021-07-19T11:58:51.388079Z"}}
train_encodings = tokenizer(train["word"].to_list(), truncation=True, padding=True)
dev_encodings = tokenizer(dev["word"].to_list(), truncation=True, padding=True)
test_encodings = tokenizer(test["word"].to_list(), truncation=True, padding=True)
train_labels = label_encoder.transform(train["label"].to_list())
dev_labels = label_encoder.transform(dev["label"].to_list())
test_labels = label_encoder.transform(test["label"].to_list())
print(dev_labels)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-19T11:58:53.267437Z","iopub.execute_input":"2021-07-19T11:58:53.267789Z","iopub.status.idle":"2021-07-19T11:58:53.27407Z","shell.execute_reply.started":"2021-07-19T11:58:53.267759Z","shell.execute_reply":"2021-07-19T11:58:53.273004Z"}}
print(len(train_encodings['input_ids']))
print(train_labels)
print(test_labels)
print(dev_labels)
# print(train["word"].to_list())

# %% [code]
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


train_dataset = HebrewNERDataset(train_encodings, train_labels)
dev_dataset = HebrewNERDataset(dev_encodings, dev_labels)
test_dataset = HebrewNERDataset(test_encodings, test_labels)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T14:46:12.04107Z","iopub.execute_input":"2021-07-17T14:46:12.043824Z","iopub.status.idle":"2021-07-17T14:46:12.060645Z","shell.execute_reply.started":"2021-07-17T14:46:12.043774Z","shell.execute_reply":"2021-07-17T14:46:12.059324Z"}}
train_dataset.__getitem__(2)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T14:46:13.726625Z","iopub.execute_input":"2021-07-17T14:46:13.726986Z","iopub.status.idle":"2021-07-17T14:46:19.678281Z","shell.execute_reply.started":"2021-07-17T14:46:13.726955Z","shell.execute_reply":"2021-07-17T14:46:19.677312Z"}}


# %% [code]
CUDA_LAUNCH_BLOCKING = 1
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    logging_steps=10
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=dev_dataset  # evaluation dataset
)

trainer.train()
trainer.save_model("./alephbert_ner")

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:03:40.047085Z","iopub.execute_input":"2021-07-17T15:03:40.047423Z","iopub.status.idle":"2021-07-17T15:03:40.771479Z","shell.execute_reply.started":"2021-07-17T15:03:40.047393Z","shell.execute_reply":"2021-07-17T15:03:40.770532Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:03:53.757575Z","iopub.execute_input":"2021-07-17T15:03:53.758064Z","iopub.status.idle":"2021-07-17T15:04:05.965373Z","shell.execute_reply.started":"2021-07-17T15:03:53.758029Z","shell.execute_reply":"2021-07-17T15:04:05.964418Z"}}
raw_pred, _, _ = trainer.predict(test_dataset)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:03:50.580511Z","iopub.execute_input":"2021-07-17T15:03:50.580899Z","iopub.status.idle":"2021-07-17T15:03:50.592569Z","shell.execute_reply.started":"2021-07-17T15:03:50.580865Z","shell.execute_reply":"2021-07-17T15:03:50.588848Z"}}
import numpy as np

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)


# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T08:42:43.08002Z","iopub.execute_input":"2021-07-17T08:42:43.080355Z","iopub.status.idle":"2021-07-17T08:42:43.088822Z","shell.execute_reply.started":"2021-07-17T08:42:43.080322Z","shell.execute_reply":"2021-07-17T08:42:43.087491Z"}}
y_pred

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T07:47:08.707508Z","iopub.execute_input":"2021-07-17T07:47:08.707861Z","iopub.status.idle":"2021-07-17T07:47:08.720714Z","shell.execute_reply.started":"2021-07-17T07:47:08.707826Z","shell.execute_reply":"2021-07-17T07:47:08.719682Z"}}
test_dataset.labels

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:04:08.336533Z","iopub.execute_input":"2021-07-17T15:04:08.33692Z","iopub.status.idle":"2021-07-17T15:04:08.432179Z","shell.execute_reply.started":"2021-07-17T15:04:08.336887Z","shell.execute_reply":"2021-07-17T15:04:08.430992Z"}}
from sklearn.metrics import f1_score, recall_score, precision_score

count_equals = 0
for a, b in zip(test_dataset.labels, y_pred):
    if a == b:
        count_equals += 1
print("accuracy=" + str(count_equals / len(y_pred)))


def evaluate(y_test, predicted):
    print("Recall Macro: " + str(recall_score(y_test, predicted, average='macro')))
    print("Precision Macro: " + str(precision_score(y_test, predicted, average='macro')))
    print("F1 Macro: " + str(f1_score(y_test, predicted, average='macro')))
    print("Recall Micro: " + str(recall_score(y_test, predicted, average='micro')))
    print("Precision Micro: " + str(precision_score(y_test, predicted, average='micro')))
    print("F1 Micro: " + str(f1_score(y_test, predicted, average='micro')))
    print("F1: " + str(f1_score(y_test, predicted, average='weighted')))


evaluate(test_dataset.labels, y_pred)
res = (list(filter(lambda x: x[1] != 61, list(zip(list(test_dataset.labels), list(y_pred))))))
test_no_o, pred_no_o = list(zip(*res))
print("F1: " + str(f1_score(test_no_o, pred_no_o, average='micro')))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:04:14.429911Z","iopub.execute_input":"2021-07-17T15:04:14.430234Z","iopub.status.idle":"2021-07-17T15:04:14.905132Z","shell.execute_reply.started":"2021-07-17T15:04:14.430204Z","shell.execute_reply":"2021-07-17T15:04:14.90432Z"}}
test_sent = "אלברט איינשטיין נולד ב גרמניה ו גר ב שווייץ"
test_sent = test_sent.split(" ")
test_sent = tokenizer(test_sent, truncation=True, padding=True)
test_sent = HebrewNERDataset(test_sent, [0, 0, 0, 0, 0, 0])
test_sent_pred, _, _ = trainer.predict(test_sent)
test_sent_pred = np.argmax(test_sent_pred, axis=1)
print(test_sent)
print(label_encoder.inverse_transform(test_sent_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:11:18.215907Z","iopub.execute_input":"2021-07-17T15:11:18.216227Z","iopub.status.idle":"2021-07-17T15:11:18.686283Z","shell.execute_reply.started":"2021-07-17T15:11:18.216198Z","shell.execute_reply":"2021-07-17T15:11:18.685479Z"}}
test_s = "אלברט איינשטיין נולד בגרמניה בחודש מרץ בשנת 1879 והיה למדען המפורסם שזכה בפרס נובל"
# test_s = "אלברט איינשטיין נולד ב גרמניה ו גר ב שווייץ"
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:04:26.41412Z","iopub.execute_input":"2021-07-17T15:04:26.414433Z","iopub.status.idle":"2021-07-17T15:04:26.885386Z","shell.execute_reply.started":"2021-07-17T15:04:26.414403Z","shell.execute_reply":"2021-07-17T15:04:26.884603Z"}}
test_s = "אלברט איינשטיין זכה בפרס נובל"
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:04:28.786615Z","iopub.execute_input":"2021-07-17T15:04:28.786958Z","iopub.status.idle":"2021-07-17T15:04:29.26459Z","shell.execute_reply.started":"2021-07-17T15:04:28.786929Z","shell.execute_reply":"2021-07-17T15:04:29.263733Z"}}
test_s = 'אברהם נדל יסד את חברת אגד ועבד כנהג אוטובוס בישראל'
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:09:27.314584Z","iopub.execute_input":"2021-07-17T15:09:27.314955Z","iopub.status.idle":"2021-07-17T15:09:27.821761Z","shell.execute_reply.started":"2021-07-17T15:09:27.314924Z","shell.execute_reply":"2021-07-17T15:09:27.82098Z"}}
test_s = 'מיכאל אנדה היה סופר ילדים גרמני'
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:04:33.310345Z","iopub.execute_input":"2021-07-17T15:04:33.310707Z","iopub.status.idle":"2021-07-17T15:04:33.789913Z","shell.execute_reply.started":"2021-07-17T15:04:33.310659Z","shell.execute_reply":"2021-07-17T15:04:33.789095Z"}}
test_s = "אריאנה גרנדה היא זמרת חמודה"
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:11:40.748109Z","iopub.execute_input":"2021-07-17T15:11:40.74864Z","iopub.status.idle":"2021-07-17T15:11:41.358964Z","shell.execute_reply.started":"2021-07-17T15:11:40.748582Z","shell.execute_reply":"2021-07-17T15:11:41.358126Z"}}
test_s = "יוסי שטיינמץ הוא נגן קלרינט"
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:12:12.244861Z","iopub.execute_input":"2021-07-17T15:12:12.24519Z","iopub.status.idle":"2021-07-17T15:12:12.71718Z","shell.execute_reply.started":"2021-07-17T15:12:12.24516Z","shell.execute_reply":"2021-07-17T15:12:12.716234Z"}}
test_s = "מלצרית אמרה ל מנהל טבח הלך"
test_s = test_s.split(" ")
test_s_tokenized = tokenizer(test_s, truncation=True, padding=True)
test_s_dataset = HebrewNERDataset(test_s_tokenized, [0 for i in range(len(test_s))])
test_s_pred, _, _ = trainer.predict(test_s_dataset)
test_s_pred = np.argmax(test_s_pred, axis=1)
print(test_s)
print(label_encoder.inverse_transform(test_s_pred))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:06:08.610255Z","iopub.execute_input":"2021-07-17T15:06:08.610585Z","iopub.status.idle":"2021-07-17T15:06:34.197618Z","shell.execute_reply.started":"2021-07-17T15:06:08.610555Z","shell.execute_reply":"2021-07-17T15:06:34.196818Z"}}
import shutil

shutil.make_archive('alephbert_ner_occ_morph', 'zip', './alephbert_ner')

# %% [code] {"execution":{"iopub.status.busy":"2021-07-17T15:06:46.900196Z","iopub.execute_input":"2021-07-17T15:06:46.900515Z","iopub.status.idle":"2021-07-17T15:06:46.907651Z","shell.execute_reply.started":"2021-07-17T15:06:46.900486Z","shell.execute_reply":"2021-07-17T15:06:46.90675Z"}}
from IPython.display import FileLink

FileLink('./alephbert_ner_occ_morph.zip')
