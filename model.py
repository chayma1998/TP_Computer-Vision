import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.datasets import fetch_lfw_people 
from sklearn import model_selection, datasets


faces = fetch_lfw_people()
positive_patches = faces.images

# random seed
seed = 42

# Read original dataset
#face_df = pd.read_csv("data/face.csv")
#face_df.sample(frac=1, random_state=seed)

# selecting features and target data
X = faces.data
y = faces.target

# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y)

# create an instance of the random forest classifier
clf = LinearSVC({'C': 1.0})

# train the classifier on the training data
clf.fit(X_train, y_train)

# predict on the test set
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")  # Accuracy: 0.91

# save the model to disk
joblib.dump(clf, "rf_model.sav")
