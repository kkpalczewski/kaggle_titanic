# data analysis
import pandas as pd
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity

# download all data
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

new_data = data_train_test_all
#new_data['Age'] = np.log(new_data['Age'])
#age = sns.FacetGrid(new_data, col='Pclass')
#age.map(plt.hist, 'Age', bins=15)

new_data = pd.DataFrame(data_train_test_all, columns = ['Pclass','Age'])
new_data["Age"] = new_data["Age"].apply(lambda x: 0 if math.isnan(x) else x)
new_data = new_data[new_data["Age"] != 0]

class1 = new_data[new_data['Pclass'] == 1]
class2 = new_data[new_data['Pclass'] == 2]
class3 = new_data[new_data['Pclass'] == 3]

tt = np.random.rand(len(class1)) < 0.8
class1_train = class1[tt]
class1_test = class1[~tt]

tt = np.random.rand(len(class2)) < 0.8
class2_train = class2[tt]
class2_test = class2[~tt]

tt = np.random.rand(len(class3)) < 0.8
class3_train = class3[tt]
class3_test = class3[~tt]

prob_sets = [[class1_train,class1_test],[class2_train,class2_test],[class3_train,class3_test]]
prob_kernels = ['gaussian','tophat']
kde = [0,0,0]
i = 0
for class_prob in prob_sets:
    act_error = 10e6
    for kernel in prob_kernels:
        new_kde = KernelDensity(kernel=kernel, bandwidth=0.2)
        new_kde.fit(class_prob[0])
        new_error = np.sum(np.matmul(new_kde.score_samples(class_prob[1]),new_kde.score_samples(class_prob[1])))
        if new_error < act_error:
            kde[i] = new_kde
            act_error = new_error
    i += 1
i = 0
data_train_test_all["Age"] = data_train_test_all["Age"].apply(lambda x: 0 if math.isnan(x) else x)
data_train_test_all["Age"] = data_train_test_all.apply(lambda row: kde[row["Pclass"]-1].sample()[0][1] if row["Age"] == 0 else row["Age"], axis = 1)

#print(data_train_test_all.describe())
#plt.hist(data_train_test_all["Age"], bins=10)
#plt.show()

table_check = pd.merge(data_train_labels, data_train_test_all, on=['PassengerId','PassengerId'])

table_fare = pd.DataFrame(table_check, columns = ['Fare', 'Survived'])
table_age = pd.DataFrame(table_check, columns = ['Age', 'Survived'])

table_fare['Fare'] = pd.qcut(table_fare['Fare'], 6)
print(table_fare.groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True))
data_train_test_all.loc[data_train_test_all['Fare'] <= 7.775, 'Fare'] = 0
data_train_test_all.loc[data_train_test_all['Fare'] <= 8.662, 'Fare'] = 1
data_train_test_all.loc[data_train_test_all['Fare'] <= 14.454, 'Fare'] = 2
data_train_test_all.loc[data_train_test_all['Fare'] <= 26.0, 'Fare'] = 3
data_train_test_all.loc[data_train_test_all['Fare'] <= 52.369, 'Fare'] = 4
data_train_test_all.loc[data_train_test_all['Fare'] > 52.369, 'Fare'] = 5
data_train_test_all['Fare'] = data_train_test_all['Fare'].astype(int)

table_age['Age'] = pd.qcut(table_age['Age'], 6)
print(table_age.groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True))
data_train_test_all.loc[data_train_test_all['Age'] <= 17.822, 'Age'] = 0
data_train_test_all.loc[data_train_test_all['Age'] <= 22.0, 'Age'] = 1
data_train_test_all.loc[data_train_test_all['Age'] <= 28.0, 'Age'] = 2
data_train_test_all.loc[data_train_test_all['Age'] <= 36.868, 'Age'] = 3
data_train_test_all.loc[data_train_test_all['Age'] <= 43.0, 'Age'] = 4
data_train_test_all.loc[data_train_test_all['Age'] > 43, 'Age'] = 5
data_train_test_all['Age'] = data_train_test_all['Age'].astype(int)

test_data = data_train_test_all[data_train_test_all["trainOrTest"] == 'test']
train_data = data_train_test_all[data_train_test_all["trainOrTest"] == 'train']
train_data = pd.merge(train_data, data_train_labels, on=['PassengerId','PassengerId'])

test_data = test_data.drop(["trainOrTest"], axis=1)
train_data = train_data.drop(["trainOrTest"], axis=1)

test_data.to_csv('.\\..\\data\\test.csv', sep='\t', encoding='utf-8')
train_data.to_csv('.\\..\\data\\train.csv', sep='\t', encoding='utf-8')

