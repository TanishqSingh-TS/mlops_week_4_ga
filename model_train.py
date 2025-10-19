Project_Id = "keen-phalanx-473718-p1"
Location = "us-central1"  
Bucket_URI = "gs://mlops-course-keen-phalanx-473718-p1"

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from google.cloud import aiplatform
from datetime import datetime
import joblib
from zoneinfo import ZoneInfo
from sklearn import metrics



data = pd.read_csv("./data/iris.csv")
train, test = train_test_split(data, test_size = 0.2, stratify = data['species'], random_state = 55)

X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species


model = DecisionTreeClassifier(max_depth = 3, random_state = 1)
model.fit(X_train,y_train)


train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))

model_stored_path = "artifacts"

os.makedirs(model_stored_path, exist_ok=True)

joblib.dump(model, f'{model_stored_path}/model.joblib')

with open(f'{model_stored_path}/metrics.txt', 'w') as f:
        f.write(f"Train Accuracy: {train_accuracy}\n")
        f.write(f"Test Accuracy: {test_accuracy}\n")
        f.write(f"Model Stored Path: {model_stored_path}\n")




    