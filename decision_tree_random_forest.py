# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os
import graphviz

# Loading & Preprocessing the Data
# Load the dataset
df = pd.read_csv("heart.csv")

# Split features and target (assuming 'target' is the label column)
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier and Visualize

# Train
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Predict & Evaluate
y_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Visualize
dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns, class_names=['No Disease', 'Disease'],
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format='png', cleanup=False)

# Analyzing Overfitting & Controlling Tree Depth

for depth in [1, 2, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Depth {depth}: Train Acc = {train_acc:.2f}, Test Acc = {test_acc:.2f}")

# Training a Random Forest & Comparing its Accuracy

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")

# Interpreting Feature Importances

importances = rf.feature_importances_
feat_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_names, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()

# Cross-Validation for Robust Evaluation

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")