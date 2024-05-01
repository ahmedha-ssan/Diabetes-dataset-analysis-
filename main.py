from sklearn.preprocessing import LabelEncoder
import math
import random
import pandas as pd
from Naive import NaiveBayes ,Naiveaccuracy_score
from DecisionTree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


filename = './diabetes_prediction_dataset.csv'
data = pd.read_csv(filename)
data.drop(['age', 'smoking_history', 'bmi', 'blood_glucose_level','HbA1c_level'], axis=1, inplace=True)
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
#print(data.head(100))

percent = float(100)
num_rows = len(data)
records_to_read = int(percent / 100 * num_rows)
print(records_to_read)
data = data[:records_to_read] # select first rows

X = data.drop([data.columns[-1]], axis=1)
y = data[data.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print('Total number of examples:', len(data))
print('Training examples:', len(X_train))
print('Test examples:', len(X_test))




#######################################################
# Train the NAIVE model
Naiveclassifier = NaiveBayes()
Naiveclassifier.fit(X_train, y_train)

test_accuracy = accuracy_score(y_test, Naiveclassifier.predict(X_test))
#print(nb_clf.predict(X_test))
print("Test Accuracy: {}".format(test_accuracy))


test_accuracy = Naiveaccuracy_score(y_test, Naiveclassifier.predict(X_test))
#print(nb_clf.predict(X_test))
print("Test Accuracy: {}".format(test_accuracy))

# user_data = pd.DataFrame({'gender': [0], 'hypertension': [0], 'heart_disease': [1]})
# prediction = Naiveclassifier.predict(user_data)
# print("Predicted diabetes:", prediction[0])



#######################################################
# Train the DT model
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

DTclassifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
DTclassifier.fit(X_train,Y_train)

print('Total number of examples:', len(data))
print('Training examples:', len(X_train))
print('Test examples:', len(X_test))

Y_pred = DTclassifier.predict(X_test) 

DTACC = accuracy_score(Y_test, Y_pred)*100
print('Accuracy of the DT model:', DTACC)

