# Import libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App heading
st.write("""
# Explore different ML models and datasets
Let see which one is best?
""")

# Puting dataset name in box and then box in sidebar
dataset_name = st.sidebar.selectbox(
    "Select a dataset",
    ("Iris", "Breast Cancer", "Wine")
)

# Putting classifier name in box and also in sidebar
classifier_name = st.sidebar.selectbox(
    "Select a classifier",
    ("KNN", "SVM", "Random Foreset")
)

# Define function to load dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y
# Call function puting a variable
X, y = get_dataset(dataset_name)

# Print shape of data on app
st.write("Shape of dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))

# Putting different classifier parameters as input
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10)
        params["C"] = C  #its the degree of correct classification
    elif classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K   # its the number of nearest neighbour
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth    # depth of every tree that grow in random forest
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators   # number of trees
    return params

# Call function and put equal to variable
params = add_parameter_ui(classifier_name)

# Define classifier base on classifier_name and params
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == "SVM":
        clf = SVC(C=params["C"])
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)
        
    return clf

# Call the function and put equal to variable
clf = get_classifier(classifier_name, params)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Train the model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Test the accuracy score of the model and print it
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy =", acc)

### PLOT DATASET ###
# Draw all the features in 2 dimensional by using pca
pca = PCA(2)
X_projected = pca.fit_transform(X)

# Slice data in 0 and 1 forms
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, 
            c=y, alpha=0.8,
            cmap="viridis")

plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

# Plt.show()
st.pyplot(fig)