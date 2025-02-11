import pandas as pd
import numpy as np
import random
import os

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer, BertModel
import torch


os.chdir(r"D:\CUHK\FTEC4008\hw1")  # set your working directory

# Version 2: Using SVM with BERT Embedding

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


# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to extract BERT embeddings
def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the mean of the last hidden state
    last_hidden_state = outputs.last_hidden_state
    sentence_embedding = last_hidden_state.mean(dim=1).squeeze()
    return sentence_embedding.numpy()

# Prepare X (features) and y (labels)
X = np.array([get_bert_embedding(text, model, tokenizer) for text in data['text']])
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

