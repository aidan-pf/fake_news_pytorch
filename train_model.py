import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy

#Get training data from https://huggingface.co/datasets/GonzaloA/fake_news

#CSV path of training data
csv_to_train = '/Users/aidan/projects/Python/crime/fake_news/train.csv'

# load the 'title' and 'text' columns from the CSV file into separate Pandas series
df_title = pd.read_csv(csv_to_train, usecols=['title','label'])
df_text = pd.read_csv(csv_to_train, usecols=['text','label'])

# preprocess the 'title' data
# tokenize the 'title'
df_title['title_tokens'] = df_title['title'].apply(lambda x: nltk.word_tokenize(x))
# lowercase the 'title'
df_title['title_tokens'] = df_title['title_tokens'].apply(lambda x: [w.lower() for w in x])
# remove punctuation from the 'title'
df_title['title_tokens'] = df_title['title_tokens'].apply(lambda x: [w for w in x if re.match('^[a-zA-Z]+$', w)])

# preprocess the 'text' data
# tokenize the 'text'
df_text['text_tokens'] = df_text['text'].apply(lambda x: nltk.word_tokenize(x))
# lowercase the 'text'
df_text['text_tokens'] = df_text['text_tokens'].apply(lambda x: [w.lower() for w in x])
# remove punctuation from the 'text'
df_text['text_tokens'] = df_text['text_tokens'].apply(lambda x: [w for w in x if re.match('^[a-zA-Z]+$', w)])

# concatenate the list of tokens from the 'title' and 'text' columns into a single string
df_title['title_tokens'] = df_title['title_tokens'].apply(lambda x: ' '.join(x))
df_text['text_tokens'] = df_text['text_tokens'].apply(lambda x: ' '.join(x))

# create a bag-of-words representation of the 'title' and 'text' data
vectorizer = CountVectorizer()
vectors_title = vectorizer.fit_transform(df_title['title_tokens'].tolist())
vectors_text = vectorizer.fit_transform(df_text['text_tokens'].tolist())

# split the 'title' and 'text' data into training and testing sets
X_train_title, X_test_title, y_train, y_test = train_test_split(vectors_title, df_title['label'], test_size=0.2, stratify=df_title['label'])
X_train_text, X_test_text, y_train, y_test = train_test_split(vectors_text, df_text['label'], test_size=0.2, stratify=df_text['label'])

# concatenate the 'title' and 'text' data into a single matrix
X_train = scipy.sparse.hstack((X_train_title, X_train_text))
X_test = scipy.sparse.hstack((X_test_title, X_test_text))

# convert the string labels to integers using the LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# define the PyTorch model
class SentimentModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = SentimentModel(input_dim1=vectors_title.shape[1], input_dim2=vectors_text.shape[1], hidden_dim=100, output_dim=2)

# train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(20):
    inputs = torch.from_numpy(X_train.todense()).float()
    labels = torch.from_numpy(y_train).long()  # convert to long tensor

    # forward pass
    outputs = model(inputs)

    # compute the loss
    loss = criterion(outputs, labels)

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch {}: loss = {:.4f}'.format(epoch, loss.item()))

# evaluate the model on the test data
inputs = torch.from_numpy(X_test.todense()).float()
labels = torch.from_numpy(y_test).long()  # convert to long tensor
outputs = model(inputs)
_, predicted = torch.max(outputs, 1)

correct = (predicted == labels).sum().item()
total = len(labels)
accuracy = correct / total

print(f'Test accuracy: {accuracy:.4f}')


