import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    #recursive function to build the tree
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split and best_split.get("info_gain", 0) > 0:  # Check if best_split is not empty
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    #function to find the best split    
    def get_best_split(self, dataset, num_samples, num_features):
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

def train_test_split(X, y, random_state=None, test_size=0.25):
    # Get number of samples
    n_samples = X.shape[0]
    # Set the seed for the random number generator
    np.random.seed(random_state)
    # Shuffle the indices
    shuffled_indices = np.random.permutation(np.arange(n_samples))
    # Determine the size of the test set
    test_size = int(n_samples * test_size)
    # Split the indices into test and train
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    # Split the features and target arrays into test and train
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test
    

def accuracy(y_true, y_pred):
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 

# filename = './diabetes_prediction_dataset.csv'
# data = pd.read_csv(filename)
# data.drop(['age', 'bmi', 'blood_glucose_level','HbA1c_level'], axis=1, inplace=True)
# label_encoder = LabelEncoder()
# data['gender'] = label_encoder.fit_transform(data['gender'])
# data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])
    
# percent = float(10)
# num_rows = len(data)
# records_to_read = int(percent / 100 * num_rows)
# data = data[:records_to_read]

# X = data.iloc[:, :-1].values
# Y = data.iloc[:, -1].values.reshape(-1,1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.25)
# classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=39999999999999999999999999999999)
# classifier.fit(X_train,Y_train)

# print(accuracy(Y_test, classifier.predict(X_test))*100)

# X_pred = X_test[:, [0, 1, 2, 3]]
# predictions = classifier.predict(X_pred)
# predictions_df = pd.DataFrame({
#     'Gender': X_pred[:, 0],
#     'Hypertension': X_pred[:, 1],
#     'Heart Disease': X_pred[:, 2],
#     'Smoking History': X_pred[:, 3],
#     'Diabetes Prediction': predictions
#     })
# # print(X_test)
# # print(predictions_df)
# #print(np.unique(predictions))
# # count_0 = 0
# # count_1 = 0

# # # Iterate through the predictions
# # for prediction in predictions:
# #     if prediction == 0:
# #         count_0 += 1
# #     elif prediction == 1:
# #         count_1 += 1

# # # Print the counts
# # print("Number of 0s:", count_0)
# # print("Number of 1s:", count_1)