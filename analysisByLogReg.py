import pandas as pd
import numpy as np
import random
import os
from gensim.models import KeyedVectors

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Version 1 @author Tian Jiayan

os.chdir(r"D:\CUHK\FTEC4008\hw1")  # set your working directory

# Random seed setting for ensuring reproduction
# Please don't remove the random seed
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
file_path = "fintech_data.csv"
data = pd.read_csv(file_path, encoding="latin1", header=None)
data.columns = ['sentiment', 'text']

# uncomment to check if file is correctly read
#  print(data)

# Load the pretrained word embedding of GloVe representation

# Remark: a file of higher dimension can be used, but requires more computing resources
glove_path = "glove.6B.50d.txt"
glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
word = "finance"
word_embedding = glove_model[word]  # 50-dimensional representation of "finance"


# print(word_embedding)

# Function to convert text into average word embedding
def text_to_embedding(text, model, vector_size=50):
    words = text.split()  # Tokenize text
    word_vectors = [model[word] for word in words if word in model]  # Extract word embeddings
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)  # Average word vectors
    else:
        return np.zeros(vector_size)  # Return zero vector if no valid words


# Preparing data
X = np.array([text_to_embedding(text, glove_model) for text in data["text"]])
# Create a label encoder to transform text labels into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["sentiment"].values)

# The follow code blocks can be used to save the output X to avoid repeated training
# in case the training time is long
'''
np.save("embedded_text.npy", X)
print("Embeddings saved successfully!")
X = np.load("embedded_text.npy")
print("Embeddings loaded successfully!")
'''

# Split dataset (80% train, 20% test). The split outputs follow fixed order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

losses = []


def store_loss_on_iteration(classifier, X_train, y_train):
    """ Capture the loss at each iteration """
    loss = classifier.loss_curve_  # This gives us the loss curve of the training process
    losses.append(loss)


# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=5000, multi_class='multinomial', solver='lbfgs')

# Train the model
model.fit(X_train, y_train)


# Predict the sentiment for the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
print(classification_report(y_test, y_pred))

