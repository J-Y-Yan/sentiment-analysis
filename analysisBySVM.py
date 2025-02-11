import pandas as pd
import numpy as np
import random
import os
from gensim.models import KeyedVectors

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

os.chdir(r"D:\CUHK\FTEC4008\hw1")  # set your working directory

# Version 1: Using SVM with GloVe Embedding

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

# Split dataset (80% train, 20% test). The split outputs follow fixed order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Initialize the SVM classifier
svm_model = SVC(kernel='linear', random_state=random_seed)

# Train the SVM model
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model using accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

# Print the evaluation results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
'''
   @:param target_names Changes the class label from (0, 1, 2) to (Negative, Neutral, Positive)
'''
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

