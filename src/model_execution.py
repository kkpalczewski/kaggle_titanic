import pandas as pd
from xgboost import DMatrix, XGBClassifier, cv
from xgboost import plot_importance, plot_tree
import datetime
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

PREDICTORS = ['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = 'Survived'
ID_COLUMN = 'PassengerId'


def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy (Train): %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

class XgbMod():
    def __init__(self, params=None):
        if params is None:
            self.params = {"n_estimators": 20, "seed": 41, 'max_depth': 8}
            self.classifier = XGBClassifier(self.params)
        else:
            self.params = params
            self.classifier = XGBClassifier(**self.params)

    def trainModel(self, data_train):
        modelfit(self.classifier, data_train, PREDICTORS, TARGET, useTrainCV=True, cv_folds=5, early_stopping_rounds=50)

    def predict(self, data_test_X, data_test_y=None):
        self.data_test_X = data_test_X
        self.Y_pred = self.classifier.predict(data_test_X[PREDICTORS])
        self.Y_pred = pd.DataFrame(data={'Survived': self.Y_pred})
        if data_test_y is not None:
            print("Classifier accuracy score on TEST SET - XGboost - ", str(metrics.accuracy_score(self.Y_pred, data_test_y)))
        return self.Y_pred

    def plotTraining(self):
        plot_importance(self.classifier)
        plot_tree(self.classifier)
        plt.show()

    def savePredictionCSV(self, out_dir):
        time_stamp = datetime.datetime.now().strftime('__%Y_%m_%d__%H_%M_%S.csv')
        out = self.data_test_X.join(self.Y_pred)
        out = pd.DataFrame(out, columns=[ID_COLUMN, TARGET])
        out.to_csv(out_dir + time_stamp, sep=',', encoding='utf-8', index=False)

    def makeCVsearch(self, params, data_train):
        cvsearch = GridSearchCV(estimator=xgbm.classifier, param_grid=params, scoring='roc_auc', n_jobs=4, iid=False,
                             cv=5)
        cvsearch.fit(data_train[PREDICTORS], data_train[TARGET])
        for score in cvsearch.grid_scores_:
            print(score)
        print('-' * 30)
        print(cvsearch.best_params_)
        print('-' * 30)
        print(cvsearch.best_score_)


if __name__=='__main__':
    data_train = pd.read_csv(filepath_or_buffer="..\\data\\train_without_intervals.csv", delimiter='\t')
    data_train_y = pd.DataFrame(data_train[TARGET])
    data_train_X = data_train.drop([TARGET], axis=1)
    data_test_X = pd.read_csv("..\\data\\test_without_intervals.csv", delimiter='\t')
    split_X_train, split_X_test, split_y_train, split_y_test = train_test_split(data_train_X, data_train_y, test_size=0.2, random_state=42)

    #first stap -
    params = {"n_estimators": 1000,
              "seed": 27,
              "max_depth": 2,
              "min_child_weight": 2,
              "gamma": 0.3,
              "colsample_bytree": 0.65,
              "subsample": 0.4,
              "reg_alpha": 0
              }

    # params = {"n_estimators": 1000,
    #           "seed": 42,
    #           "max_depth": 5,
    #           "min_child_weight": 8,
    #           "gamma": 0.4,
    #           "colsample_bytree": 0.9,
    #           "subsample": 0.7,
    #           "reg_alpha": 0}
    xgbm = XgbMod(params)
    xgbm.trainModel(data_train)
    xgbm.predict(data_test_X)

    #param_test_1 = {'max_depth': list(range(6, 12, 1)), 'min_child_weight': list(range(6, 10, 1))}
    #param_test_2 = {'gamma':[i/20.0 for i in range(2,8)]}
    #param_test_3 = {'subsample': [i / 10.0 for i in range(1, 10)], 'colsample_bytree': [i / 10.0 for i in range(1, 10)]}
    #param_test_4 = {'subsample': [i / 100.0 for i in range(25, 35)], 'colsample_bytree': [i / 100.0 for i in range(45, 55)]}
    #param_test_5 = {'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100]}
    #param_test_6 = {'reg_alpha': [0, 1e-10, 1e-6, 1e-5, 1e-4, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    #param_test_7 = {'learning_rate': [0.01,0.03,0.01,0.03]}
    #xgbm.makeCVsearch(param_test_1,data_train)

    xgbm.plotTraining()

    xgbm.savePredictionCSV("..\\analysis\\upload_8_xgb")

