import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load your dataset from data.pickle
with open("data.pickle", "rb") as f:
    dataset = pickle.load(f)

X = dataset["data"]
y = dataset["labels"]

# Optionally split data for evaluation (you can also train on all data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model (optional)
score = clf.score(X_test, y_test)
print("Test accuracy:", score)

# Save the trained model to model.p
with open("model.p", "wb") as f:
    pickle.dump({"model": clf}, f)

print("✅ Model saved to model.p!")
