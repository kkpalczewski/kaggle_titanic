
# data analysis
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import itertools

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

download_data_train = pd.read_csv("..\download\\train.csv")
download_data_test = pd.read_csv("..\download\\test.csv")

#create new labels
data_train_labels = pd.DataFrame(download_data_train, columns = ['PassengerId', 'Survived'])

#create one table for easier data preparation
download_data_train["trainOrTest"] = "train"
download_data_test["trainOrTest"] = "test"
download_data_train_sub = download_data_train.drop(["Survived"], axis=1)
data_train_test_all = pd.concat([download_data_train_sub, download_data_test])

data_train_test_all["Fare"] = data_train_test_all["Fare"].apply(lambda x: x if not math.isnan(x) else np.mean(data_train_test_all['Fare']))
data_train_test_all["Sex"] = data_train_test_all["Sex"].apply(lambda x: 1 if x == "male" else 0)
data_train_test_all["Embarked"] = data_train_test_all["Embarked"].apply(lambda x: 2 if x == "C" else (1 if x == 'Q' else 0))

data_train_test_all['Title'] = data_train_test_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
data_train_test_all['Title'] = data_train_test_all['Title'].replace('Mlle', 'Miss')
data_train_test_all['Title'] = data_train_test_all['Title'].replace('Ms', 'Miss')
data_train_test_all['Title'] = data_train_test_all['Title'].replace('Mme', 'Mrs')
data_train_test_all['Title'] = data_train_test_all['Title'].apply(lambda x: x if x in ['Master', 'Miss', 'Mr', 'Mrs'] else 'Rare')

title_map = {'Mrs': 4, 'Miss': 3, 'Master': 2, 'Mr' : 1, 'Rare' : 0}
data_train_test_all['Title'] = data_train_test_all['Title'].map(title_map)

data_train_test_all = data_train_test_all.drop(["Ticket"], axis=1)
data_train_test_all = data_train_test_all.drop(["Cabin"], axis=1)
data_train_test_all = data_train_test_all.drop(["Name"], axis=1)



# No special correlation
corr_mat = data_train_test_all.corr()
fig, ax = plt.subplots(figsize=(20,12))
sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax)
plt.show()

class ModelAge():
    def __init__(self, train_set_X, train_set_y, test_set_X, test_set_y):
        self.train_set_X = train_set_X
        self.train_set_y = train_set_y
        self.test_set_X = test_set_X
        self.test_set_y = test_set_y

        self.features = [tf.feature_column.numeric_column(key='Pclass'),
                         tf.feature_column.numeric_column(key='Sex'),
                         tf.feature_column.numeric_column(key='SibSp'),
                         tf.feature_column.numeric_column(key='Parch'),
                         tf.feature_column.numeric_column(key='Fare'),
                         tf.feature_column.numeric_column(key='Embarked'),
                         tf.feature_column.numeric_column(key='Title'),
                         ]

    def buildRegressor(self):
        self.regressor = tf.estimator.DNNRegressor(feature_columns=self.features,
                                                   hidden_units=[7,6],
                                                   model_dir = "/tmp/titanic")
    def trainRegressor(self, num_epochs, steps):
        input_fn = tf.estimator.inputs.pandas_input_fn(x=self.train_set_X,
                                                       y=self.train_set_y,
                                                       num_epochs=num_epochs,
                                                       shuffle = False)
        self.regressor.train(input_fn, steps=steps)

    def evaluateRegressor(self, num_epochs):
        input_fn = tf.estimator.inputs.pandas_input_fn(x=self.test_set_X,
                                                       y=self.test_set_y,
                                                       num_epochs=num_epochs,
                                                       shuffle = False)
        ev = self.regressor.evaluate(input_fn=input_fn)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))

    def predictRegressor(self, prediction_set, num_epochs):
        input_fn = tf.estimator.inputs.pandas_input_fn(x=prediction_set,
                                                       y=None,
                                                       num_epochs=num_epochs,
                                                       shuffle = False)
        y = self.regressor.predict(input_fn=input_fn)
        return y

    def predictScores(self, prediction_set, num_epochs):
        input_fn = tf.estimator.inputs.pandas_input_fn(x=prediction_set,
                                                       y=None,
                                                       num_epochs=num_epochs,
                                                       shuffle = False)
        y = self.regressor.predict(input_fn=input_fn)
        return y

if __name__ == '__main__':
    all_data = data_train_test_all
    all_data = all_data.drop(['trainOrTest','PassengerId'], axis=1)
    all_data["Age"] = all_data["Age"].apply(lambda x: 0 if math.isnan(x) else x)
    prediction_set = all_data[all_data["Age"] == 0]
    test_train_set = all_data[all_data["Age"] != 0]
    tt = np.random.rand(len(test_train_set)) < 0.8

    train_data = test_train_set[tt]
    test_data = test_train_set[~tt]

    train_data_y = train_data["Age"]
    train_data_X = train_data.drop(["Age"], axis = 1)

    test_data_y = test_data["Age"]
    test_data_X = test_data.drop(["Age"], axis = 1)
    #print(train_data_y)
    NewModel = ModelAge(train_data_X, train_data_y, test_data_X, test_data_y)
    NewModel.buildRegressor()
    NewModel.trainRegressor(10, 5000)
    NewModel.evaluateRegressor(10)
    y = NewModel.predictRegressor(prediction_set, 10)
    z = NewModel.predictScores(prediction_set, 10)
    # for itemy in y:
    #     print(itemy['predictions'])
    # for itemz in z:
    #     print(itemz['predictions'])

    test_X = NewModel.predictRegressor(test_data_X, 10)
    test_y = list(test_data_y)
    i = 0
    sum = 0
    for item in test_X:
        err = math.sqrt((item['predictions'][0] - test_y[i])*(item['predictions'][0] - test_y[i]))
        print(err)
        sum += err
        i += 1
        print(i)
    print(err/len(test_X))