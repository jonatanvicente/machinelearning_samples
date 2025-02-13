import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Features
y = (X > 5).astype(int).ravel()  # Labels: 1 if X > 5, else 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual data', alpha=0.6, marker='o')
plt.scatter(X_test, y_pred, color='red', label='Predicted data', alpha=0.6, marker='x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression')
plt.legend()
plt.show()