import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

import seaborn as sns


train_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'
test_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\data\xvector_pretrained_dataset_with_embeddings.csv'

train_df = pd.read_csv(train_file, low_memory=False)
test_df = pd.read_csv(test_file, low_memory=False)


drop_cols = ['file','speaker','label','speaker_encoded','SpkId','length','gender']
X_train = train_df.drop(drop_cols, axis=1)
y_train = train_df['label_coding']
X_test = test_df.drop(drop_cols, axis=1)
y_test = test_df['label_coding']


scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 64),  
                        activation='relu',
                        solver='adam',
                        max_iter=500,
                        random_state=42)
mlp_clf.fit(X_train_scaled, y_train)


y_train_pred = mlp_clf.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")


y_pred = mlp_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


report_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\reports\MLP_classification_report.txt'
with open(report_file, 'w') as f:
    f.write(f"Accuracy: {accuracy:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Spoof (0)', 'Bona-fide (1)'],
            yticklabels=['Spoof (0)', 'Bona-fide (1)'])
plt.title("MLP Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

conf_matrix_file = r'C:\Users\R I B\Desktop\Study\Project Laboratory 2\Deep-Speaker-Embeddings-for-Robust-Detection-of-Deepfake-Speech\results\MLP_confusion_matrix.png'
plt.savefig(conf_matrix_file, dpi=300)
plt.show()
