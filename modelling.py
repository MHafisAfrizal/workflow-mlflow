import mlflow
import os
import pandas as pd
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Tracking
print("Tracking URI:", mlflow.get_tracking_uri())
print(os.getcwd())
print(os.listdir())

# Setup direktori
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'namadataset_preprocessing')

# Muat data
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()

# Inisialisasi experiment dan autolog
mlflow.set_experiment("Iris_Classification")
mlflow.sklearn.autolog()

# Jalankan dan log run
with mlflow.start_run():
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    print("Akurasi:", accuracy)
    print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred))

    print("Run ID:", mlflow.active_run().info.run_id)
    print("Run saved in:", mlflow.get_artifact_uri())