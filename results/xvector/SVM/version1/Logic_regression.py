
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler


# Load the training data
train_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'
train_df = pd.read_csv(train_file, low_memory=False)

# Load the testing data
test_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'
test_df = pd.read_csv(test_file, low_memory=False)

# Separate features and target in training data
X_train = train_df.drop(['file','speaker','label','speaker_encoded','SpkId','length','gender'], axis=1)  # Features
y_train = train_df['label_coding']  # Target

# Separate features and target in testing data
X_test = test_df.drop(['file','speaker','label','speaker_encoded','SpkId','length','gender'], axis=1)  # Features
y_test = test_df['label_coding']  # Target



# Apply Robust Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression Classifier
logreg_clf = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter if convergence issues occur
logreg_clf.fit(X_train_scaled, y_train)

# Accuracy of train set
y_train_pred = logreg_clf.predict(X_train_scaled)

# Evaluate accuracy on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")


# Make predictions on the test set
y_pred = logreg_clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


report_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\reports\LRclassification_report_meta.txt'
with open(report_file, 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot and save the confusion matrix as an image
plt.figure(figsize=(10, 8))  # High resolution
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Spoof (0)', 'Bona-fide (1)'], yticklabels=['Spoof (0)', 'Bona-fide (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Save the confusion matrix plot
confusion_matrix_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\results\LRconfusion_matrix_meta.png'
plt.savefig(confusion_matrix_file, dpi=300)  # Save with high resolution (300 DPI)
plt.show()