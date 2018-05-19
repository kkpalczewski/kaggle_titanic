import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import datetime
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt

class RanForMod():
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
    def trainModel(self, train_X, train_y):
        self.train_X_dropped = train_X.drop(['PassengerId'], axis=1)
        self.classifier.fit(self.train_X_dropped, train_y)
    def predict(self, test_X, test_y = None):
        self.test_X_dropped = test_X.drop(['PassengerId'], axis=1)
        self.test_X = test_X
        self.Y_pred = self.classifier.predict(self.test_X_dropped)
        self.Y_pred = pd.DataFrame(data={'Survived':self.Y_pred})
        if test_y is not None:
            print("Classifier score - RandomForest - ", str(self.classifier.score(self.test_X_dropped, test_y)))
        return self.Y_pred
    def savePredictionCSV(self, out_dir):
        time_stamp = datetime.datetime.now().strftime('__%Y_%m_%d__%H_%M_%S.csv')
        out = self.test_X.join(self.Y_pred)
        out = pd.DataFrame(out, columns=['PassengerId', 'Survived'])
        out.to_csv(out_dir+time_stamp, sep=',', encoding='utf-8', index=False)

if __name__=='__main__':
    data_train = pd.read_csv(filepath_or_buffer="..\\data\\train.csv", delimiter='\t')
    data_train_y = np.asarray(data_train['Survived'])
    data_train_X = data_train.drop(['Survived'], axis=1)
    data_test_X = pd.read_csv("..\\data\\test.csv", delimiter='\t')
    X_train, X_test, y_train, y_test = train_test_split(data_train_X, data_train_y, test_size=0.2, random_state=42)

    #Random Forest classifier
    rfm = RanForMod()
    rfm.trainModel(data_train_X, data_train_y)
    rfm.predict(data_test_X)
    #rfm.savePredictionCSV("..\\analysis\\upload_2")


    train_X_dropped = data_train_X.drop(['PassengerId'], axis=1)
    test_X_dropped = data_test_X.drop(['PassengerId'], axis=1)
    X_train = X_train.drop(['PassengerId'], axis=1)
    X_test = X_test.drop(['PassengerId'], axis=1)

    xgb_model = xgb.XGBClassifier(n_estimators=20, seed=41, max_depth=8)
    xgb_model.fit(X_train, y_train)

    print(xgb_model.score(X_test, y_test))

    #pred = xgb_model.predict(X_test)
    #pred = pd.DataFrame(pred, columns=['Survived'])
    #outcome = data_test_X.join(pred)
    #outcome = pd.DataFrame(outcome, columns=['PassengerId', 'Survived'])
    #outcome.to_csv("..\\analysis\\upload_3_xgb.csv", sep=',', encoding='utf-8', index=False)

    xgb.plot_importance(xgb_model)
    plt.show()
    xgb.plot_tree(xgb_model)
    plt.show()
    print(pred)