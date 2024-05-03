import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def Naiveaccuracy_score(y_true, y_pred):
    """Calculate the accuracy score."""
    return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)



class NaiveBayes:
    def __init__(self):
        """Initialize the Naive Bayes classifier."""
        self.features = []
        self.likelihoods = {}
        self.class_priors = {}
        self.pred_priors = {}
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.train_size = 0
        self.num_feats = 0

    def fit(self, X, y):
        """Fit the classifier to the data."""
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}
            self.pred_priors[feature] = {}

            for feat_val in np.unique(self.X_train[feature]):
                self.pred_priors[feature][feat_val] = 0

                for outcome in np.unique(self.y_train):
                    self.likelihoods[feature][str(feat_val) + '_' + str(outcome)] = 0
                    self.class_priors[outcome] = 0

        self._calc_class_prior()
        self._calc_likelihoods()
        self._calc_predictor_prior()

    def _calc_class_prior(self):
        """Calculate prior probabilities of classes."""
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self):
        """Calculate likelihoods."""
        for feature in self.features:
            for outcome in np.unique(self.y_train):
                outcome_count = sum(self.y_train == outcome)
                feat_likelihood = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].value_counts().to_dict()
                for feat_val, count in feat_likelihood.items():
                    self.likelihoods[feature][str(feat_val) + '_' + str(outcome)] = count / outcome_count

    def _calc_predictor_prior(self):
        """Calculate prior probabilities of features."""
        for feature in self.features:
            feat_vals = self.X_train[feature].value_counts().to_dict()
            for feat_val, count in feat_vals.items():
                self.pred_priors[feature][feat_val] = count / self.train_size

    def predict(self, X):
        """Predict the class labels for the input data."""
        results = []
        X = np.array(X)

        for query in X:
            probs_outcome = {}
            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                evidence = 1

                for feat, feat_val in zip(self.features, query):
                    likelihood *= self.likelihoods[feat][str(feat_val) + '_' + str(outcome)]
                    evidence *= self.pred_priors[feat][feat_val]

                posterior = (likelihood * prior) / evidence
                probs_outcome[outcome] = posterior

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)

def NAIVEtrain_test_split(X, y, test_size=0.25, random_state=None):
    """Custom implementation of train-test split."""
    if random_state:
        np.random.seed(random_state)
    
    num_rows = len(X)
    indices = np.arange(num_rows)
    np.random.shuffle(indices)
    
    test_indices = indices[:int(test_size * num_rows)]
    train_indices = indices[int(test_size * num_rows):]
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

# data = pd.read_csv("diabetes_prediction_dataset.csv")
# data.drop(['age', 'smoking_history', 'bmi', 'blood_glucose_level','HbA1c_level'], axis=1, inplace=True)
# label_encoder = LabelEncoder()
# data['gender'] = label_encoder.fit_transform(data['gender'])

# percent = float(1)
# num_rows = len(data)
# records_to_read = int(percent / 100 * num_rows)
# data = data[:records_to_read]

# X = data.drop([data.columns[-1]], axis=1)
# y = data[data.columns[-1]]

# X_train, X_test, y_train, y_test = NAIVEtrain_test_split(X, y, test_size=0.25, random_state=42)
# # for index, row in enumerate(X_test.values):
# #     print("Index:", index, "Row:", row)

# print(X_train.head())

# Naiveclassifier = NaiveBayes()
# Naiveclassifier.fit(X_train, y_train)
# test_accuracy = Naiveaccuracy_score(y_test, Naiveclassifier.predict(X_test))

# print("Test Accuracy: {}".format(test_accuracy))
    

    

# X_pred = X_test[['gender', 'hypertension', 'heart_disease']]
# predictions = Naiveclassifier.predict(X_pred)
# predictions_df = pd.DataFrame({
#     'gender': X_pred['gender'],
#     'hypertension': X_pred['hypertension'],
#     'heart_disease': X_pred['heart_disease'],
#     'diabetes': predictions
    
# })
# #print(np.unique(predictions))
# #print(predictions_df)
