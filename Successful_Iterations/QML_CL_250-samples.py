### GoEmotions QML Pipeline (Modified to use HuggingFace Datasets)

# Step 1: Import dependencies
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.pandas()


# Step 2: Load GoEmotions dataset
print("Loading GoEmotions dataset from HuggingFace...")
dataset = load_dataset("go_emotions")

# Using only a sample for faster prototyping (optional)
sample_size = 250
train_data = dataset["train"].shuffle(seed=42).select(range(sample_size))
df = pd.DataFrame({
    "text": train_data["text"],
    "label": [labels[0] if len(labels) > 0 else -1 for labels in train_data["labels"]]
})
df = df[df["label"] != -1]  # remove -1 labels
print("Dataset shape:", df.shape)
df.head()


# Step 3: Load tokenizer and BERT model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")


# Step 4: Tokenize and generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

print("Generating BERT embeddings...")
X = np.vstack(train_data["text"][:]).tolist()
X_embeddings = [get_bert_embedding(text) for text in tqdm(X)]


# Step 5: Prepare labels
# Convert multi-label to single-label (choose the first label for simplicity)
y = [labels[0] if len(labels) > 0 else -1 for labels in train_data["labels"]]

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# Remove samples with -1 labels (no label)
X_embeddings = [emb for emb, label in zip(X_embeddings, y) if label != -1]
y = [label for label in y if label != -1]


# Step 6: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)


# Step 7: Train classical SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


# Step 8: Predictions
y_pred = svm.predict(X_test)


# Step 9: Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Step 10: Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# More Plots 

### Per-Class Accuracy Bar Plot
correct_preds = (y_pred == y_test)
class_accuracy = {}

for i, label in enumerate(np.unique(y_test)):
    idx = y_test == label
    class_accuracy[label_encoder.inverse_transform([label])[0]] = np.mean(correct_preds[idx])

plt.figure(figsize=(12, 6))
sns.barplot(x=list(class_accuracy.keys()), y=list(class_accuracy.values()))
plt.ylabel("Accuracy")
plt.xlabel("Class Label")
plt.title("Per-Class Accuracy")
plt.xticks(rotation=45)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

### Macro-Average Metrics
from sklearn.metrics import precision_score, recall_score, f1_score

macro_precision = precision_score(y_test, y_pred, average='macro')
macro_recall = recall_score(y_test, y_pred, average='macro')
macro_f1 = f1_score(y_test, y_pred, average='macro')

metrics = {'Precision': macro_precision, 'Recall': macro_recall, 'F1 Score': macro_f1}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
plt.ylim(0, 1.0)
plt.title("Macro-Average Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

### Prediction Distribution
unique, counts = np.unique(y_pred, return_counts=True)
pred_dist = dict(zip(label_encoder.inverse_transform(unique), counts))

plt.figure(figsize=(12, 5))
sns.barplot(x=list(pred_dist.keys()), y=list(pred_dist.values()))
plt.xlabel("Predicted Class")
plt.ylabel("Frequency")
plt.title("Prediction Distribution Across Classes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


