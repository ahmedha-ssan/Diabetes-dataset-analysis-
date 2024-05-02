import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Naive import NaiveBayes, Naiveaccuracy_score
from DecisionTree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def browse_file():
    filename = filedialog.askopenfilename(initialdir="/D:/ass/4th year/DATA MINING/DiabetesClassification/", title="Select file",
                                          filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    if filename:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, filename)

def process_data():
    file_path = entry_path.get()
    if not file_path:
        messagebox.showerror("Error", "Please select a CSV file.")
        return
    
    try:
        percent = float(entry_percent.get())
        if percent <= 0 or percent > 100:
            raise ValueError("Percentage must be between 0 and 100.")
    except ValueError:
        messagebox.showerror("Error", "Invalid percentage value.")
        return
    
    try:
        data = pd.read_csv(file_path)
        data.drop(['age', 'smoking_history', 'bmi', 'blood_glucose_level','HbA1c_level'], axis=1, inplace=True)
        label_encoder = LabelEncoder()
        data['gender'] = label_encoder.fit_transform(data['gender'])

        num_rows = len(data)
        records_to_read = int(percent / 100 * num_rows)
        data = data[:records_to_read]

        X = data.drop([data.columns[-1]], axis=1)
        y = data[data.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        Naiveclassifier = NaiveBayes()
        Naiveclassifier.fit(X_train, y_train)
        Naiveaccuracy = Naiveaccuracy_score(y_test, Naiveclassifier.predict(X_test))
        text_output.insert(tk.END, f"Test Accuracy of NAIVE : {Naiveaccuracy}\n\n")
        text_output.insert(tk.END, f"# of test set : {len(X_test)}\n\n")
        
        X_pred = X_test[['gender', 'hypertension', 'heart_disease']]
        predictions = Naiveclassifier.predict(X_pred)
        predictions_df = pd.DataFrame({
            'gender': X_pred['gender'],
            'hypertension': X_pred['hypertension'],
            'heart_disease': X_pred['heart_disease'],
            'diabetes': predictions
        })
        print(np.unique(predictions))
        text_output.insert(tk.END, "Predictions of NAIVE model:\n")
        text_output.insert(tk.END, predictions_df.to_string(index=False) + '\n\n')

        X_train, X_test, Y_train, Y_test = train_test_split(X.values, y.values.reshape(-1,1), test_size=0.25, random_state=0)
        DTclassifier = DecisionTreeClassifier(min_samples_split=3, max_depth=99900000000)
        DTclassifier.fit(X_train,Y_train)

        Y_pred = DTclassifier.predict(X_test) 
        DTaccuracy = accuracy_score(Y_test, Y_pred)*100
        text_output.insert(tk.END, f"Accuracy of the DT model: {DTaccuracy}\n\n")
        text_output.insert(tk.END, f"# of test set : {len(X_test)}\n\n")
        X_pred = X_test[:, [0, 1, 2]] 
        predictions = DTclassifier.predict(X_pred)
        predictions_df = pd.DataFrame({
            'Gender': X_test[:, 0],
            'Hypertension': X_test[:, 1],
            'Heart_Disease': X_test[:, 2],
            'Diabetes_Prediction': predictions
        })
        print("DT")
        print(np.unique(predictions))
        text_output.insert(tk.END, "Predictions of DT model:\n")
        text_output.insert(tk.END, predictions_df.to_string(index=False) + '\n\n')
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create main window
root = tk.Tk()
root.title("Diabetes Prediction")

# Frame for inputs
frame_input = tk.Frame(root)
frame_input.pack(padx=10, pady=10)

label_path = tk.Label(frame_input, text="CSV File Path:")
label_path.grid(row=0, column=0, padx=5, pady=5)

entry_path = tk.Entry(frame_input, width=50)
entry_path.grid(row=0, column=1, padx=5, pady=5)

button_browse = tk.Button(frame_input, text="Browse", command=browse_file)
button_browse.grid(row=0, column=2, padx=5, pady=5)

label_percent = tk.Label(frame_input, text="Percentage of Data to Read:")
label_percent.grid(row=1, column=0, padx=5, pady=5)

entry_percent = tk.Entry(frame_input)
entry_percent.grid(row=1, column=1, padx=5, pady=5)

# Frame for output
frame_output = tk.Frame(root)
frame_output.pack(padx=10, pady=10)

text_output = tk.Text(frame_output, width=80, height=20)
text_output.pack(padx=5, pady=5)

button_process = tk.Button(root, text="Process Data", command=process_data)
button_process.pack(pady=10)

root.mainloop()

# from sklearn.preprocessing import LabelEncoder
# import math
# import random
# import pandas as pd
# from Naive import NaiveBayes ,Naiveaccuracy_score
# from DecisionTree import DecisionTreeClassifier

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# filename = './diabetes_prediction_dataset.csv'
# data = pd.read_csv(filename)
# data.drop(['age', 'smoking_history', 'bmi', 'blood_glucose_level','HbA1c_level'], axis=1, inplace=True)
# label_encoder = LabelEncoder()
# data['gender'] = label_encoder.fit_transform(data['gender'])
# #print(data.head(100))

# percent = float(100)
# num_rows = len(data)
# records_to_read = int(percent / 100 * num_rows)
# #print(records_to_read)
# data = data[:records_to_read] # select first rows

# X = data.drop([data.columns[-1]], axis=1)
# y = data[data.columns[-1]]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# print('Total number of examples:', len(data))
# print('Training examples:', len(X_train))
# print('Test examples:', len(X_test))




# #######################################################
# # Train the NAIVE model
# Naiveclassifier = NaiveBayes()
# Naiveclassifier.fit(X_train, y_train)
# Naiveaccuracy = Naiveaccuracy_score(y_test, Naiveclassifier.predict(X_test))
# #print(test_accuracy.predict(X_test))
# print("Test Accuracy of NAIVE : {}".format(Naiveaccuracy))


# X_pred = X_test[['gender', 'hypertension', 'heart_disease']]
# predictions = Naiveclassifier.predict(X_pred)
# predictions_df = pd.DataFrame({
#     'gender': X_pred['gender'],
#     'hypertension': X_pred['hypertension'],
#     'heart_disease': X_pred['heart_disease'],
#     'diabetes': predictions
# })
# print(predictions_df)


# #######################################################
# # Train the DT model
# X = data.iloc[:, :-1].values
# Y = data.iloc[:, -1].values.reshape(-1,1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# DTclassifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
# DTclassifier.fit(X_train,Y_train)

# Y_pred = DTclassifier.predict(X_test) 
# DTaccuracy = accuracy_score(Y_test, Y_pred)*100
# print('Accuracy of the DT model:', DTaccuracy)


# X_pred = X_test[:, [0, 1, 2]] 
# predictions = DTclassifier.predict(X_pred)
# predictions_df = pd.DataFrame({
#     'Gender': X_test[:, 0],  # Assuming age is the first column
#     'Hypertension': X_test[:, 1],
#     'Heart_Disease': X_test[:, 2],
#     'Diabetes_Prediction': predictions
#     })
# print(len(predictions_df))
# print(predictions_df)