import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
 
# Load datasets
train_df = pd.read_csv(r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector\pretrain_XVector_train_dataset_with_embeddings.csv')
test_df = pd.read_csv(r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector\pretrain_XVector_test_dataset_with_embeddings.csv')
 
# Separate metadata from embeddings
meta_columns = ['file', 'speaker', 'label', 'speaker_encoded', 'SpkId', 'length', 'gender', 'label_coding']
X_train = train_df.drop(columns=meta_columns).values
y_train = train_df['label_coding'].values
 
X_test = test_df.drop(columns=meta_columns).values
y_test = test_df['label_coding'].values
 
# Ensure no overlap between training and testing speakers
assert not set(train_df['SpkId']).intersection(set(test_df['SpkId']))
 
# Apply Standard Scaling (recommended for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Apply Min-Max Scaling
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
 
 
# Train SVM Model (Linear Kernel)
svm = SVC(kernel='linear', probability=True, random_state=42)
model = CalibratedClassifierCV(svm)  # Enables probability estimates for LinearSVC
model.fit(X_train_scaled, y_train)
 
# Model Evaluation & Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability for class 1 (Bona-fide)
 
# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
 
# Compute Per-Class Accuracy (Avoid division by zero)
class_accuracies = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
 
# Generate Classification Report
report = classification_report(y_test, y_pred, digits=4)
 
# Save Classification Report & Per-Class Accuracy to TXT File
report_filename = 'classification_report_svM_robustscaling_resnet.txt'
with open(report_filename, 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nPer-Class Accuracy:\n")
 
    class_labels = ['0: Spoof', '1: Bona-fide']
    for i, acc in enumerate(class_accuracies):
        f.write(f"Accuracy for class {class_labels[i]}: {acc:.4f}\n")
 
print(f"Classification report saved as {report_filename}")
 
# Plot and Save Confusion Matrix as High-Quality Image
plt.figure(figsize=(8, 6), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
 
plt.title("Confusion Matrix (SVM - wavlm)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
 
conf_matrix_filename = "confusion_matrix_svm_robustscaling_resnet.png"
plt.savefig(conf_matrix_filename, dpi=300, bbox_inches="tight")
plt.close()
 
print(f"Confusion matrix saved as {conf_matrix_filename}")
 
# Save Predictions and Probabilities to CSV
probabilities_filename = "predictions_probabilities_svm_robustscaling_resnet.csv"
df_results = pd.DataFrame({
    'File': test_df['file'],
    "Actual_Label": y_test,
    "Predicted_Label": y_pred,
    "Probability_BonaFide": y_prob
})
 
df_results.to_csv(probabilities_filename, index=False)
 
print(f"Prediction probabilities saved as {probabilities_filename}")