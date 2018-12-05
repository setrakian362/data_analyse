
import pandas as pd


## Importing the datasets
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

## saving passenger id in advance in order to submit later.
passengerid = test.PassengerId
## We will drop PassengerID and Ticket since it will be useless for our data.
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

print (train.info())
print ("*"*40)
print (test.info())


total = train.isnull().sum().sort_values(ascending = False)
percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])


percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True)*100,2))
## creating a df with the #
total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
## concating percent and total dataframe
pd.concat([total, percent], axis = 1, keys=['Total_per_group', 'Percent'])

corr = train.corr()**2
corr.Survived.sort_values(ascending=False)
