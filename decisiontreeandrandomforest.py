import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv("churn-bigml-80.xls")

#convert manually
data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})
data['Churn'] = data['Churn'].astype(int)  # Convert to binary 0/1

# transform the dataset to 0 and 1
data = pd.get_dummies(data, columns=['State'], drop_first=True)

# Churn is the column that is to be predicted
X = data.drop('Churn', axis=1)
y = data['Churn']

#random state is a value that will give same results even after running multiple times
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

dt_model = DecisionTreeClassifier(random_state=20)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

rf_model = RandomForestClassifier(random_state=20, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate the models
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
