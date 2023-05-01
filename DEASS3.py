from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load the covtype dataset
covtype = fetch_covtype()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(covtype.data, covtype.target, test_size=0.2, random_state=42)

# Initialize a Random Forest classifier with default parameters
rfc = RandomForestClassifier()

# Get the default parameters and their values
params = rfc.get_params()

# Print the parameters and their values
for key, value in params.items():
    print(key, "=", value)

# Measure the time taken to train the classifier
start_time = time.time()
rfc.fit(X_train, y_train)
end_time = time.time()
time_taken_train = end_time - start_time
print("Time taken to train the classifier:", time_taken_train, "seconds")

# Use the trained classifier to make predictions on the testing set and measure the time taken
start_time = time.time()
y_pred = rfc.predict(X_test)
end_time = time.time()
time_taken_pred = end_time - start_time
print("Time taken to make predictions on the testing set:", time_taken_pred, "seconds")

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
