import pandas as pd
from sklearn.svm import SVC  
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load training and testing data
train_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'
test_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'

train_df = pd.read_csv(train_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)

# Define features and target
drop_cols = ['file','speaker','label','speaker_encoded','SpkId','length','gender']
X_train = train_df.drop(drop_cols, axis=1)
y_train = train_df['label_coding']
X_test = test_df.drop(drop_cols, axis=1)
y_test = test_df['label_coding']

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train SVM
svm_clf = SVC(kernel='linear', C=1.0, random_state=42)  # You can change kernel='rbf' if needed
svm_clf.fit(X_train_scaled, y_train)

# Evaluate on training set
y_train_pred = svm_clf.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Predict and evaluate on test set
y_pred = svm_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save classification report
report_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\reports\SVM_classification_report.txt'
with open(report_file, 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Spoof (0)', 'Bona-fide (1)'],
            yticklabels=['Spoof (0)', 'Bona-fide (1)'])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Save confusion matrix
confusion_matrix_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\results\SVM_confusion_matrix.png'
plt.savefig(confusion_matrix_file, dpi=300)
plt.show()
