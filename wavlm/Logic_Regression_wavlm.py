import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
 
# Load datasets
train_df = pd.read_csv(r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\wavlm\pretrain_Wavlm_train_dataset_with_embeddings.csv')
test_df = pd.read_csv(r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\wavlm\pretrain_Wavlm_test_dataset_with_embeddings.csv')
 
# Separate metadata from embeddings
meta_columns = ['file', 'speaker', 'label', 'speaker_encoded', 'SpkId', 'length', 'gender', 'label_coding']
X_train = train_df.drop(columns=meta_columns).values
y_train = train_df['label_coding'].values
 
X_test = test_df.drop(columns=meta_columns).values
y_test = test_df['label_coding'].values
 
# Ensure no overlap between training and testing speakers
assert not set(train_df['SpkId']).intersection(set(test_df['SpkId']))
 
#Apply feature scaling
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
 
# Apply Min-Max Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
 
# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
 
# Model Evaluation & Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
 
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
 
# Compute per-class accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
 
# Get classification report
report = classification_report(y_test, y_pred, digits=4)
 
# Save classification report & per-class accuracy to a text file
report_filename = 'classification_report_resnet_.txt'
with open(report_filename, 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nPer-Class Accuracy:\n")
    class_labels = ['0: Spoof', '1: Bona-fide']
    for i, acc in enumerate(class_accuracies):
        f.write(f"Accuracy for class {class_labels[i]}: {acc:.4f}\n")
 
print(f"Classification report saved as {report_filename}")
 
# Confusion Matrix Visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues')
 
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig('confusion_matrix__resnet_Scaler.png', dpi=300)
plt.show()
 
# Save predictions and probabilities
results_df = pd.DataFrame({
    'File': test_df['file'],
    'Actual_Label': y_test,
    'Predicted_Label': y_pred,
    'Probability': y_prob
})
results_df.to_csv('predictions_probabilities_fine_tuned_mlp_resnet.csv', index=False)