# data analysis
import pandas as pd
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity

PREDICTORS = ['Pclass', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
TARGET = 'Survived'
ID_COLUMN = 'PassengerId'

# download all data
download_data_train = pd.read_csv("..\download\\train.csv")
download_data_test = pd.read_csv("..\download\\test.csv")

#create new labels
data_train_labels = pd.DataFrame(download_data_train, columns = [ID_COLUMN, TARGET])

#create one table for easier data preparation
download_data_train["trainOrTest"] = "train"
download_data_test["trainOrTest"] = "test"
download_data_train_sub = download_data_train.drop([TARGET], axis=1)
data_all = pd.concat([download_data_train_sub, download_data_test])

################################################ DATA REPLACING ###################################################################
data_all["Fare"] = data_all["Fare"].apply(lambda x: x if not math.isnan(x) else np.mean(data_all['Fare']))

data_all["Sex"] = data_all["Sex"].apply(lambda x: 1 if x == "male" else 0)

data_all["Embarked"] = data_all["Embarked"].apply(lambda x: 2 if x == "C" else (1 if x == 'Q' else 0))

################################################ DATA EXTRACTION ##################################################################
data_all['Title'] = data_all.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
data_all['Title'] = data_all['Title'].replace('Mlle', 'Miss')
data_all['Title'] = data_all['Title'].replace('Ms', 'Miss')
data_all['Title'] = data_all['Title'].replace('Mme', 'Mrs')
data_all['Title'] = data_all['Title'].apply(lambda x: x if x in ['Master', 'Miss', 'Mr', 'Mrs'] else 'Rare')
title_map = {'Mrs': 4, 'Miss': 3, 'Master': 2, 'Mr' : 1, 'Rare' : 0}
data_all['Title'] = data_all['Title'].map(title_map)

############################################### DATA DROPP #########################################################################
data_all = data_all.drop(["Ticket"], axis=1)

data_all = data_all.drop(["Cabin"], axis=1)

data_all = data_all.drop(["Name"], axis=1)

############################################## FILL DATA ##########################################################################
new_data = data_all
#new_data['Age'] = np.log(new_data['Age'])
#age = sns.FacetGrid(new_data, col='Pclass')
#age.map(plt.hist, 'Age', bins=15)

new_data = pd.DataFrame(data_all, columns = ['Pclass','Age'])
new_data["Age"] = new_data["Age"].apply(lambda x: 0 if math.isnan(x) else x)
new_data = new_data[new_data["Age"] != 0]

class1 = new_data[new_data['Pclass'] == 1]
class2 = new_data[new_data['Pclass'] == 2]
class3 = new_data[new_data['Pclass'] == 3]

# tt = np.random.rand(len(class1)) < 0.8
# class1_train = class1[tt]
# class1_test = class1[~tt]
#
# tt = np.random.rand(len(class2)) < 0.8
# class2_train = class2[tt]
# class2_test = class2[~tt]
#
# tt = np.random.rand(len(class3)) < 0.8
# class3_train = class3[tt]
# class3_test = class3[~tt]
#
# prob_sets = [[class1_train,class1_test],[class2_train,class2_test],[class3_train,class3_test]]
# prob_kernels = ['gaussian','tophat']
# kde = [0,0,0]
# i = 0
# for class_prob in prob_sets:
#     act_error = 10e6
#     for kernel in prob_kernels:
#         new_kde = KernelDensity(kernel=kernel, bandwidth=0.2)
#         new_kde.fit(class_prob[0])
#         new_error = np.sum(np.matmul(new_kde.score_samples(class_prob[1]),new_kde.score_samples(class_prob[1])))
#         if new_error < act_error:
#             kde[i] = new_kde
#             act_error = new_error
#     i += 1
kde = [0,0,0]
i = 0
for class_prob in [class1, class2, class3]:
    act_error = 10e6
    new_kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
    new_kde.fit(class_prob)
    kde[i] = new_kde
    i += 1
data_all["Age"] = data_all["Age"].apply(lambda x: 0 if math.isnan(x) else x)
data_all["Age"] = data_all.apply(lambda row: kde[row["Pclass"]-1].sample()[0][1] if row["Age"] == 0 else row["Age"], axis = 1)

#print(data_all.describe())
#plt.hist(data_all["Age"], bins=10)
#plt.show()

#################################################### CREATE INTERVALS ##############################################################
# table_check = pd.merge(data_train_labels, data_all, on=['PassengerId','PassengerId'])
# table_fare = pd.DataFrame(table_check, columns = ['Fare', 'Survived'])
# table_age = pd.DataFrame(table_check, columns = ['Age', 'Survived'])
#
# table_fare['Fare'] = pd.qcut(table_fare['Fare'], 6)
# fares = [7.775, 8.662, 14.454, 26.0, 52.369]
# data_all.loc[data_all['Fare'] <= fares[0], 'Fare'] = 0
# for i in range(len(fares)-1):
#     data_all.loc[(data_all['Fare'] <= fares[i+1]) & (data_all['Fare'] > fares[i]), 'Fare'] = i+1
# data_all.loc[data_all['Fare'] > fares[len(fares)-1], 'Fare'] = len(fares)
data_all['Fare'] = data_all['Fare'].astype(int)

# table_age['Age'] = pd.qcut(table_age['Age'], 6)
# ages = [17.822, 22.0, 28.0, 36.868, 43.0]
# data_all.loc[data_all['Age'] <= ages[0], 'Age'] = 0
# for i in range(len(ages)-1):
#     data_all.loc[(data_all['Age'] <= ages[i+1]) & (data_all['Age'] > ages[i]), 'Age'] = i+1
# data_all.loc[data_all['Age'] > ages[len(ages)-1], 'Age'] = len(ages)
data_all['Age'] = data_all['Age'].astype(int)

test_data = data_all[data_all["trainOrTest"] == 'test']
train_data = data_all[data_all["trainOrTest"] == 'train']
train_data = pd.merge(train_data, data_train_labels, on=[ID_COLUMN, ID_COLUMN])

test_data = test_data.drop(["trainOrTest"], axis=1)
train_data = train_data.drop(["trainOrTest"], axis=1)

############################################## SAVE TO CSV ########################################################
test_data.to_csv('.\\..\\data\\test_without_intervals.csv', sep='\t', encoding='utf-8', index=False)
train_data.to_csv('.\\..\\data\\train_without_intervals.csv', sep='\t', encoding='utf-8', index=False)

if __name__ == '__main__':
    print('-'*60)
    print(train_data.describe())
    print('-'*60)
    print(test_data.describe())
