import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV

import matplotlib
matplotlib.use("agg")

data_set_dict = {"zip": ("zip.test.gz", 0),
                 "spam": ("spam.data", 57)}
data_dict = {}

for data_name, (file_name, label_col_num) in data_set_dict.items():
    data_df = pd.read_csv(file_name, sep = " ", header = None)
    data_label_vec = data_df.iloc[:, label_col_num]
    is_01 = data_label_vec.isin([0,1])
    data_01_df = data_df.loc[is_01, :]
    is_label_col = data_df.columns == label_col_num
    data_features = data_01_df.iloc[:, ~is_label_col]
    data_labels = data_01_df.iloc[:, is_label_col]
    data_dict[data_name] = (data_features, data_labels)
    #scaling the data
    n_data_features = data_features.shape[1]
    data_mean = data_features.mean().to_numpy().reshape(1, n_data_features)
    data_std = data_features.std().to_numpy().reshape(1, n_data_features)
    data_scaled = (data_features - data_mean)/data_std
    data_name_scaled = data_name + "_scaled"
    data_scaled = data_scaled.dropna(axis = "columns")
    data_dict[data_name_scaled] = (data_scaled, data_labels)
    print(data_scaled)
    #data_scaled = data_name_scaled.dropna(axis = "columns")
    #data_dict[data_name_scaled] = data_name_scaled
    

#preapring zip data for binary classification
zip_df = pd.read_csv("zip.test.gz", sep = " ", header = None)
zip_label_col_num = 0
zip_label_vec = zip_df.iloc[:, zip_label_col_num]
is_01 = zip_label_vec.isin([0,1])
zip_01_df = zip_df.loc[is_01, :]
is_label_col = zip_01_df.columns == zip_label_col_num
zip_features = zip_01_df.iloc[:, ~is_label_col]
zip_labels = zip_01_df.iloc[:, is_label_col]

#preparing spam data for binary classification
spam_df = pd.read_csv("spam.data", sep= " ", header = None)
spam_label_col_num = -1
spam_label_vec = spam_df.iloc[:, spam_label_col_num]
spam_is_01 = spam_label_vec.isin([0,1])
spam_01_df = spam_df.loc[spam_is_01, :]
spam_features = spam_df.iloc[:, :spam_label_col_num]
spam_labels = spam_df.iloc[:, spam_label_col_num]

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_dict = {
    "zip" : (zip_features, zip_labels),
    "spam" : (spam_features, spam_labels)
}

accuracy_data_frames = []
for data_name, (data_features, data_labels) in data_dict.items():
    kf = KFold(n_splits = 3, shuffle = True, random_state = 3)
    enum_obj = enumerate(kf.split(data_features))
    for fold_num, (train_index, test_index) in enum_obj:
        X_train, X_test = np.array(data_features.iloc[train_index]), np.array(data_features.iloc[test_index])
        y_train, y_test = np.ravel(data_labels.iloc[train_index]), np.ravel(data_labels.iloc[test_index])


        #K-nearest neighbors
        knn = KNeighborsClassifier()
        hp_parameters = {"n_neighbors": list(range(1,21))}
        grid = GridSearchCV(knn, hp_parameters, cv = 5)
        grid.fit(X_train, y_train)
        best_n_neighbors = grid.best_params_['n_neighbors']
        print("Best N-Neighbors = ", best_n_neighbors)
        knn = KNeighborsClassifier(n_neighbors = best_n_neighbors)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        #print(knn_pred)
        #print(knn.score(X_test, y_test))
        pipe1 = make_pipeline(StandardScaler(),\
                              KNeighborsClassifier(n_neighbors = best_n_neighbors))
        pipe1.fit(X_train, y_train)
        scaled_pred = pipe1.predict(X_test)

        #Logistic Regression
        pipe = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, max_iter=2000))
        pipe.fit(X_train, y_train)
        lr_pred = pipe.predict(X_test)
        #print(pipe.score(X_test, y_test))
        y_train_series = pd.Series(y_train)
        most_frequent_class = y_train_series.value_counts().idxmax()
        print("Most Frequent Class = ", most_frequent_class)


        #create a featureless baseline
        featureless_pred = np.full_like(y_test, most_frequent_class)

        #store predict data in dict
        pred_dict = {'nearest neighbors': knn_pred,
                     'Scaled nearest neighbors': scaled_pred,
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
total_accuracy_df = pd.concat(accuracy_data_frames, ignore_index = True)
print(total_accuracy_df)

            
import plotnine as p9

gg = p9.ggplot(total_accuracy_df, p9.aes(x ='accuracy', y = 'algorithm', fill = 'data_set'))+\
        p9.facet_grid('.~data_set') + p9.geom_point()

gg.save("Output.png",height = 10, width = 15)




#Another Version
list_of accuracy_rows = []
for data_name, (data_features, data_labels) in data_dict.items():
    kf = KFold(n_splits = 3)
    print(data_name)
    enum_obj = enumerate(kf.split(data_features))
    split_data_dict
