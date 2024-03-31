# AIML-project
import numpy as np
import pandas as pd

# Setting seed for reproducibility
np.random.seed(0)

# Number of data points
n = 1000

# Generating random data
data = {
    'PassengerId': np.arange(1, n + 1),
    'Pclass': np.random.choice([1, 2, 3], size=n),
    'Sex': np.random.choice(['male', 'female'], size=n),
    'Age': np.random.randint(1, 80, size=n),
    'SibSp': np.random.randint(0, 5, size=n),
    'Parch': np.random.randint(0, 4, size=n),
    'Fare': np.random.uniform(0, 300, size=n),
    'Survived': np.random.randint(0, 2, size=n)
}

# Creating DataFrame
df = pd.DataFrame(data)



# Convert 'Sex' to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, alpha, epochs):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= alpha * gradient
    
    return theta


# Feature matrix X
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].values

# Adding intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Target variable y
y = df['Survived'].values

# Hyperparameters
alpha = 0.01
epochs = 1000

# Training the model
theta = gradient_descent(X, y, alpha, epochs)

# Predicting probabilities
probabilities = sigmoid(np.dot(X, theta))

# Predicting survival (1 if probability >= 0.5 else 0)
predictions = (probabilities >= 0.5).astype(int)

print("Model Parameters (theta):", theta)

# Check the first few predictions
print("Predictions:", predictions[:10])

# Calculate and print accuracy
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)

# Confusion matrix
confusion_matrix = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        confusion_matrix[i, j] = np.sum((predictions == i) & (y == j))
print("Confusion Matrix:")
print(confusion_matrix)


