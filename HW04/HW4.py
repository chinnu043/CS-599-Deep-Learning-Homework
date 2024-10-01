import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid

import matplotlib

matplotlib.use("agg")


data_set_dict = {"zip": ("zip.test.gz", 0),
                 "spam": ("spam.data", 57)}
data_dict = {}

for data_name, (file_name, label_col_num) in data_set_dict.items():
    data_df = pd.read_csv(file_name, sep=" ", header=None)
    data_label_vec = data_df.iloc[:, label_col_num]
    is_01 = data_label_vec.isin([0, 1])
    data_01_df = data_df.loc[is_01, :]
    is_label_col = data_df.columns == label_col_num
    data_features = data_01_df.iloc[:, ~is_label_col]
    data_labels = data_01_df.iloc[:, is_label_col]
    data_dict[data_name] = (data_features, data_labels)
    # scaling the data
    n_data_features = data_features.shape[1]
    data_mean = data_features.mean().to_numpy().reshape(1, n_data_features)
    data_std = data_features.std().to_numpy().reshape(1, n_data_features)
    data_scaled = (data_features - data_mean) / data_std
    data_name_scaled = data_name + "_scaled"
    data_scaled = data_scaled.dropna(axis="columns")
    data_dict[data_name_scaled] = (data_scaled, data_labels)
    #print(data_scaled)
    # data_scaled = data_name_scaled.dropna(axis = "columns")
    # data_dict[data_name_scaled] = data_name_scaled

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collections import Counter


class MyKNN:
    def __init__(self, n_neighbors):
        """store n_neighbors as attribute"""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """store data"""
        self.X_train = X
        self.y_train = y


    def decision_function(self, X):
        """Compute vector of predicted scores.
        Larger values mean more likely to be in positive class."""
        scores = []
        for x in X:
            distances = np.sqrt(np.sum((x - self.X_train) ** 2, axis = 1))
            n_neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            n_neighbor_labels = [self.y_train[i] for i in n_neighbor_indices]

            most_common_label = Counter(n_neighbor_labels).most_common(1)[0][0]
            #print(most_common_label)
            scores.append(most_common_label)
        #print(scores)
        return scores

    def predict(self, X):
        return self.decision_function(X)

class MyCV:
    def __init__(self, estimator, param_grid, cv):        
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        
    def fit_one(self, param_dict, X, y):
        self.estimator.__init__(param_dict)
        self.estimator.fit(X, y)
        #print(X)
        #print(y)
     
    def fit(self, X, y):
        validation_df_list = []
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=3)
        for validation_fold, (train_index, test_index) in enumerate(kf.split(X)):
            train_data = {"X": X[train_index], "y": y[train_index]}
            test_data = {"X": X[test_index], "y": y[test_index]}

            for param_dict in self.param_grid:
                #print(param_dict)
                self.fit_one(param_dict, **train_data)
                y_pred = self.estimator.predict(test_data["X"])
                #print(y_pred)
                accuracy = np.mean(y_pred == test_data["y"])
                validation_row = pd.DataFrame({
                    "validation_fold": [validation_fold],
                    "accuracy": [accuracy],
                    "param_value": [param_dict]
                })
                validation_df_list.append(validation_row)
        validation_df = pd.concat(validation_df_list)
        best_param_dict = validation_df.groupby("param_value")["accuracy"].mean().idxmax()
        #print(best_param_dict)
        self.fit_one(best_param_dict, X, y)
        #return best_param_dict
    def predict(self, X):
        return self.estimator.predict(X)


class Featureless:
    def fit(self, X_train, y_train):
        y_train_series = pd.Series(y_train)
        self.most_freq_labels = y_train_series.value_counts().idxmax()

    def predict(self, x_test):
        test_nrow, test_ncol = x_test.shape
        return np.repeat(self.most_freq_labels, test_nrow)


accuracy_data_frames = []
for data_name, (data_features, data_labels) in data_dict.items():
    kf = KFold(n_splits=3, shuffle=True, random_state=3)
    enum_obj = enumerate(kf.split(data_features))
    for fold_num, (train_index, test_index) in enum_obj:
        X_train, X_test = np.array(data_features.iloc[train_index]), np.array(data_features.iloc[test_index])
        y_train, y_test = np.ravel(data_labels.iloc[train_index]), np.ravel(data_labels.iloc[test_index])

        # K-nearest neighbors
        knn = KNeighborsClassifier()
        hp_parameters = {"n_neighbors": list(range(1, 21))}
        grid = GridSearchCV(knn, hp_parameters, cv=5)
        grid.fit(X_train, y_train)
        best_n_neighbors = grid.best_params_['n_neighbors']
        print("Best N-Neighbors = ", best_n_neighbors)
        knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        #print(knn_pred)

        #KNN
        knn1 = MyKNN(n_neighbors = 3)
        
        #KNN + gridCV
        gridcv = MyCV(estimator = knn1, param_grid= [n_neighbors for n_neighbors in range(1,21)], cv=5)
        gridcv.fit(X_train, y_train)
        knn1_pred = gridcv.predict(X_test)
        #print(knn1_pred)

        # Logistic Regression
        pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, max_iter=2000))
        pipe.fit(X_train, y_train)
        lr_pred = pipe.predict(X_test)
        #print(lr_pred)
        # print(pipe.score(X_test, y_test))
        # y_train_series = pd.Series(y_train)
        my_learner_instance = Featureless()
        my_learner_instance.fit(X_train, y_train)
        featureless_pred = my_learner_instance.predict(X_test)
        #print(featureless_pred)
        # most_frequent_class = y_train_series.value_counts().idxmax()
        # print("Most Frequent Class = ", most_frequent_class)

        # create a featureless baseline
        # featureless_pred = np.full_like(y_test, most_frequent_class)

        # store predict data in dict
        pred_dict = {'gridSearch + nearest neighbors': knn_pred,
                     'KNN + CV': knn1_pred,
                     'linear_model': lr_pred,
                     'featureless': featureless_pred}
        test_accuracy = {}

        for algorithm, predictions in pred_dict.items():
            accuracy = accuracy_score(y_test, predictions)
            test_accuracy[algorithm] = accuracy

        for algorithm, accuracy in test_accuracy.items():
            print(f"{algorithm} Test Accuracy: {accuracy * 100}")
            accuracy_df = pd.DataFrame({
                "data_set": [data_name],
                "fold_id": [fold_num],
                "algorithm": [algorithm],
                "accuracy": [test_accuracy[algorithm]]})
            accuracy_data_frames.append(accuracy_df)
        print(f"****************************End of {data_name}({fold_num})*****************************")
total_accuracy_df = pd.concat(accuracy_data_frames, ignore_index=True)
print(total_accuracy_df)

import plotnine as p9

gg = p9.ggplot(total_accuracy_df, p9.aes(x='accuracy', y='algorithm')) + \
     p9.facet_grid('.~data_set') + p9.geom_point()

gg.save("Output.png", height=8, width=12)
