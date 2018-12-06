import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Importing the datasets
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

## saving passenger id in advance in order to submit later.
passengerid = test.PassengerId
## We will drop PassengerID and Ticket since it will be useless for our data.
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

print(train.info())
print("*" * 40)
print(test.info())

total = train.isnull().sum().sort_values(ascending=False)
percent = round(train.isnull().sum().sort_values(ascending=False) / len(train) * 100, 2)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True) * 100, 2))
## creating a df with the #
total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
## concating percent and total dataframe
pd.concat([total, percent], axis=1, keys=['Total_per_group', 'Percent'])

fig, ax = plt.subplots(figsize=(16, 12), ncols=2)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax=ax[0])
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax=ax[1])
ax1.set_title("Training Set", fontsize=18)
ax2.set_title('Test Set', fontsize=18)
fig.show()

train.Cabin.isnull().sum() / len(train.Cabin))

## Assigning all the null values as "N"
train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)

train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]

## Let's see the value counts now
pd.DataFrame(train.Cabin.value_counts())

# Part 3. Visualization and Feature Relations¶

pal = {'male': "green", 'female': "Pink"}
plt.subplots(figsize=(15, 8))
ax = sns.barplot(x="Sex",
y = "Survived",
data = train,
palette = pal,
linewidth = 2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize=25)
plt.ylabel("% of passenger survived", fontsize=15)
plt.xlabel("Sex", fontsize=15);



plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass",
            y = "Survived",
            data=train,
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right.
plt.xticks(val, labels);


plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass",
            y = "Survived",
            data=train,
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right.
plt.xticks(val, labels);


# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you.
ax=sns.kdeplot(train.Pclass[train.Survived == 0] ,
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] ,
               color='g',
               shade=True,
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels);


# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Fare", fontsize = 15)

# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 25)
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15);


pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)

g = sns.FacetGrid(train, size=5,hue="Survived", col ="Sex", margin_titles=True,
                palette=pal,)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Sex, Fare and Age", size = 25)
plt.subplots_adjust(top=0.85)



## dropping the three outliers where Fare is over $500
train = train[train.Fare < 500]
## factor plot
sns.factorplot(x = "Parch", y = "Survived", data = train,kind = "point",size = 8)
plt.title("Factorplot of Parents/Children survived", fontsize = 25)
plt.subplots_adjust(top=0.85)



train.describe()
train.describe(include =['O'])

train[['Pclass', 'Survived']].groupby("Pclass").mean().reset_index()

# Overview(Survived vs non survied)
survived_summary = train.groupby("Survived")
survived_summary.mean().reset_index()


pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))


import numpy as np

## heatmeap to see the correlation between features.
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(),
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1,
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);

train['Sex'] = train['Sex'].astype("category",ordered=True, categories = ['female','male']).cat.codes
import random
# separating male and female dataframe.
male = train[train['Sex'] == 1]
female = train[train['Sex'] == 0]

# getting 50 random sample for male and female.
import random
male_sample = random.sample(list(male['Survived']),50)
female_sample = random.sample(list(female['Survived']),50)

# Taking a sample means of survival feature from male and female
male_sample_mean = np.mean(male_sample)
female_sample_mean = np.mean(female_sample)

# Print them out
print ("Male sample mean: " + str(male_sample_mean))
print ("Female sample mean: " + str(female_sample_mean))
print ("Difference between male and female sample mean: " + str(female_sample_mean - male_sample_mean))


import scipy.stats as stats

print (stats.ttest_ind(male_sample, female_sample))
print ("This is the p-value when we break it into standard form: " + format(stats.ttest_ind(male_sample, female_sample).pvalue, '.32f'))


# Creating a new colomn with a
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]

def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a

train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)