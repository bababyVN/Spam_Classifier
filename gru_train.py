
from torch import Tensor
import tqdm
from tqdm.auto import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import numpy as np
import json
import pickle
from datasets import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
# Link the directories with google drive

# Select device for our deep learning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

"""# 2. Loading data"""

df = pd.read_csv('Data\combined_spam_dataset.csv')

# Rename columns if necessary to match the expected names
# (only do this if your CSV uses different column names)
# df.rename(columns={'text': 'cleaned_text', 'label': 'target'}, inplace=True)

# Split the dataset
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Match old function structure: extract x and y
x_train = train_df['text'].astype(str).tolist()
y_train = train_df['label'].astype(int).tolist()

x_val = val_df['text'].astype(str).tolist()
y_val = val_df['label'].astype(int).tolist()

x_test = test_df['text'].astype(str).tolist()
y_test = test_df['label'].astype(int).tolist()

# Rebuild the dataset dicts as before
train_data = [{"text": x_train[i], "label": y_train[i]} for i in range(len(x_train))]
val_data = [{"text": x_val[i], "label": y_val[i]} for i in range(len(x_val))]
test_data = [{"text": x_test[i], "label": y_test[i]} for i in range(len(x_test))]

# Convert to Hugging Face Dataset objects
train_data = Dataset.from_pandas(pd.DataFrame(data=train_data))
val_data = Dataset.from_pandas(pd.DataFrame(data=val_data))
test_data = Dataset.from_pandas(pd.DataFrame(data=test_data))

# Checking if the dataset object is working properly
train_data[0]

# Importing necessary libraries for NLP (vocabulary and tokenizer)

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Setting <unk> and <pad> for unknown words in vocabulary, and padding to make our text length equal
UNK_IDX, PAD_IDX = 0, 1
SPECIAL_SYMBOLS = ['<unk>', '<pad>']

token_transform = get_tokenizer('basic_english')

def tokenize_example(example, token_transform):
    tokens = token_transform(example['text'])
    return {"tokens": tokens}

# Tokenizing train, test and validation dataset from text to list of tokens
fn_kwargs = {
    'token_transform': token_transform
}
train_data = train_data.map(tokenize_example, fn_kwargs = fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs = fn_kwargs)
val_data = val_data.map(tokenize_example, fn_kwargs = fn_kwargs)

# Examining the tokens to see if it worked properly
train_data[2]

total_spam = sum([train_data[i]['label'] for i in range(len(train_data))])
print(total_spam / len(train_data))

# Building vocab from the tokens appearing the train dataset, min frequency of 2
vocab = build_vocab_from_iterator(
    train_data['tokens'],
    min_freq = 2,
    specials = SPECIAL_SYMBOLS,
    special_first = True
)
# Set default index to <unk> for unknown words
vocab.set_default_index(UNK_IDX)

# Checking vocab to see the example of conversion from text to token_id
vocab.get_itos()[:10]
vocab.lookup_indices(train_data[322]['tokens'])

def numericalize_example(example, vocab):
    ids = torch.LongTensor(vocab.lookup_indices(example['tokens']))
    return {"ids": ids}

fn_kwargs = {
    'vocab': vocab
}
# Convert train, test and validation dataset from tokens to id using the vocabulary
train_data = train_data.map(numericalize_example, fn_kwargs = fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs = fn_kwargs)
val_data = val_data.map(numericalize_example, fn_kwargs = fn_kwargs)

# collate_fn to get the input, target for each batch
def collate_fn(batch):
    x_batch = [torch.tensor(sample['ids']) for sample in batch]
    y_batch = torch.tensor([sample['label'] for sample in batch], dtype = torch.float)

    # Padding for each batch so that the size of the id list of each text become equal (size of the longest text in the batch)
    x_batch = nn.utils.rnn.pad_sequence(x_batch, padding_value = PAD_IDX, batch_first = True)
    x_batch = x_batch.long()

    return x_batch, y_batch

from torch.utils.data import DataLoader

# Setting data loader by batch for both train, test and validation dataset
BATCH_SIZE = 8
train_data_loader = DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    collate_fn = collate_fn,
    shuffle = True
)

test_data_loader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    collate_fn = collate_fn,
    shuffle = True
)

val_data_loader = DataLoader(
    dataset = val_data,
    batch_size = BATCH_SIZE,
    collate_fn = collate_fn,
    shuffle = True
)

class GRU_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout_rate=0.3):
        super(GRU_model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=(self.dropout_rate if self.num_layers > 1 else 0),
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embedding_dim)
        gru_output, hidden = self.gru(embedded)  # gru_output: (batch, seq_len, hidden_dim)
        output_logits = self.fc(gru_output[:, -1, :])  # take last output time step
        return output_logits.squeeze()


vocab_size = len(vocab)
embedding_dim = 200
hidden_dim = 200
num_layers = 1

model = GRU_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers
).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")

# Setting optimizer and loss functions for our training
optimizer = Adam(model.parameters(), lr = 1e-4)
criterion = nn.BCEWithLogitsLoss()

def train_fn(model, data_loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    for batch in data_loader:

        # Get the batch input, target
        x_train = batch[0].to(device)
        y_train = batch[1].to(device)

        # Setting the gradients to zero for training this batch
        optimizer.zero_grad()
        y_preds = model(x_train)

        # Computing loss
        loss = criterion(y_preds, y_train)

        # Computing gradients over the parameters of the model
        loss.backward()

        # Gradient clipping to prevent exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # The parameters are stepped by the gradient of loss w.r.t weights
        optimizer.step()
        epoch_loss += loss.item()

    # return average loss over batches
    return epoch_loss / len(data_loader)

# Evaluate_fn to get the average loss over batches for given dataset
def evaluate_fn(model, data_loader, optimizer, criterion, clip, devie):
      model.eval()
      epoch_loss = 0
      with torch.inference_mode():
          for batch in data_loader:
              # Get the batch input, target
              x_eval = batch[0].to(device)
              y_eval = batch[1].to(device)

              # Compute prediction
              y_pred = model(x_eval)

              #print(y_eval)
              #print(y_pred)
              # Compute loss value
              loss = criterion(y_pred, y_eval)

              epoch_loss += loss.item()

      # return average loss over batches
      return epoch_loss / len(data_loader)

import tqdm

n_epochs = 10
clip = 2

train_loss_val = []
val_loss_val = []

best_val_loss = float('inf')

filepath = 'Model\Gru'

for epoch in tqdm.tqdm(range(n_epochs)):
    # Calculating train loss and val loss
    train_loss = train_fn(model, train_data_loader, optimizer, criterion, clip, device)
    val_loss = evaluate_fn(model, val_data_loader, optimizer, criterion, clip, device)
    train_loss_val.append(train_loss)
    val_loss_val.append(val_loss)

    if val_loss < best_val_loss:
        # If it is the best model with smallest validation loss, save it
        best_val_loss = val_loss
        torch.save(model.state_dict(), filepath + "LSTM_model.pth")
     # Printing out the loss and perplexity value
    print(f"\nTrain loss: {train_loss: 7.3f}, Train perplexity: {np.exp(train_loss): 7.3f}")
    print(f"Validation loss: {val_loss: 7.3f}, Validation perplexity: {np.exp(val_loss): 7.3f}")

import matplotlib.pyplot as plt

# Plotting the learning curve (measured by perplexity through training and validation set)
plt.plot(range(1, 11), np.exp(train_loss_val), label = 'Train perplexity')
plt.plot(range(1, 11), np.exp(val_loss_val), label = 'Validation perplexity')
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
plt.title("Perplexity plot")
plt.legend()

import pickle
with open(filepath + "vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open(filepath + "LSTM_Model.pkl", "wb") as f:
    pickle.dump(model, f)

model = 0

with open(filepath + "LSTM_Model.pkl", "rb") as f:
    model = pickle.load(f)
with open(filepath + "vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

test_loss = evaluate_fn(model, test_data_loader, optimizer, criterion, clip, device)
print(f"Test loss: {test_loss: 7.3f}, Validation perplexity: {np.exp(test_loss): 7.3f}")

def predict_spam(sentence, model, tokenizer, vocab, device):

    tokens = tokenizer(sentence)
    tokens_id = torch.LongTensor(vocab.lookup_indices(tokens)).unsqueeze(0).to(device)

    print(tokens_id.shape)

    output_logits = model(tokens_id).item()
    prediction_prob = torch.sigmoid(torch.tensor(output_logits)).item()
    print(prediction_prob)
    pred = 1
    if prediction_prob > 0.5:
        pred = 1
    else:
        pred = 0
    return pred

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report

# Calculate the F1, precision, recall, accuracy and balanced accuracy
# For a given dataset for evaluation
def predict(data, model, threshold, device = 'cpu', data_type = 'train'):
    y_pred = []
    y_real = []
    for i, instance in enumerate(data):
        x = instance['ids']
        # If the text is null, we can use padding and extend it by 10
        if len(x) == 0:
            x = [PAD_IDX] * 10

        x = torch.LongTensor(x).unsqueeze(0).to(device)
        y = instance['target']
        #print(x.shape)
        output_logits = model(x).item()
        pred_prob = torch.sigmoid(torch.tensor(output_logits)).item()
        pred = 1
        if pred_prob > threshold:
            pred = 1
        else:
            pred = 0
        y_pred.append(pred)
        y_real.append(y)

    report = classification_report(y_real, y_pred, output_dict= True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"{data_type} datatype:")
    print("F1 score: " + str(float(f1)))
    print("Precision score: " + str(float(precision)))
    print("Recall score: " + str(float(recall)))
    print("Accuracy score: " + str(float(accuracy_score(y_real, y_pred))))
    print("Balanced accuracy score: " + str(float(balanced_accuracy_score(y_real, y_pred))))
    print()
predict(train_data, model, 0.5, device, 'train')
predict(test_data, model, 0.5, device, 'test')
predict(val_data, model, 0.5, device, 'val')

