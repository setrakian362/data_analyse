import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.extend(['PYDEVD_USE_FRAME_EVAL=NO'])

sns.set_style("darkgrid")

# hr = pd.read_csv('HR_comma_sep.csv')
# hr.head()
#
# #Identify the Data Types - Numpy
# hr_dd = pd.DataFrame(hr.dtypes,columns=['Numpy Dtype'])
#
# #Identify the unique values
# hr_dd['Nunique'] = hr.nunique()
#
# #Identify the Missing values
# hr_dd['MissingValues']=hr.isnull().sum()
#
# # Identify the count for each variable
# hr_dd['Count']=hr.count()
#
# # Identify the zero values for each variable
# hr_dd['ZeroValues']=(hr==0).sum()
#
#
# hr.describe(include=['number'])
# hr.describe(include=['object'])
#
#
# fig, axes = plt.subplots(ncols=3, figsize=(10,5))
#
# g = sns.distplot(hr['satisfaction_level'],ax=axes[0])
# g = sns.distplot(hr['last_evaluation'],ax=axes[1])
# g = sns.distplot(hr['average_montly_hours'],ax=axes[2])
#
#
#
# #distirbution of categorical columns using counter plots
# fig, axes = plt.subplots(ncols=2,figsize=(12,6))
# g = sns.countplot(hr["sales"], ax=axes[0])
# plt.setp(g.get_xticklabels(), rotation=45)
# g = sns.countplot(hr["salary"], ax=axes[1])
#
#
# #distirbution of other numerical features
# fig, axes = plt.subplots(ncols=3,figsize=(12,6))
# g = sns.countplot(hr["Work_accident"], ax=axes[0])
# g = sns.countplot(hr["promotion_last_5years"], ax=axes[1])
# g = sns.countplot(hr["left"], ax=axes[2])
#
# #distirbution of other numerical features
# fig, axes = plt.subplots(ncols=2,figsize=(12,6))
#
# g=hr['time_spend_company'].plot(kind='hist',ax=axes[0],bins=8)
# g.set_xlabel('time_spend_company')
# g.set_ylabel('count')
#
# g=hr['number_project'].plot(kind='hist',ax=axes[1],bins=6,color='lightgreen')
# g.set_xlabel('number_project')
# g.set_ylabel('count')
#
#
# g = sns.heatmap(hr.corr(),annot=True,cmap="RdYlGn")
# plt.title('correlation between different variables')
#
#
# #convert salary feature to an ordered categorical variable
# hr["salary"] = hr["salary"].astype("category",ordered=True, categories = ['low','medium','high']).cat.codes
# #drop the sales categorial non-nominal feature
# hr = hr.drop(labels=["sales"],axis = 1)
# #get a random sample of data for the PCA analysis
# hr = hr.sample(n=10000,replace=True)
# # Standardize features by removing the mean and scaling to
# # unit variance
# from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
#
# N = StandardScaler()
# N.fit(hr)
# hr_norm = N.transform(hr)
# from sklearn.decomposition import PCA
# from sklearn.manifold import Isomap
# # Linear dimensionality reduction using Singular Value Decomposition
# #of the data to project it to a lower dimensional space.
# pca = PCA(n_components=2)
# pca_representation = pca.fit_transform(hr_norm)
# df_pca = pd.DataFrame(pca_representation)
# df_pca.head(5)
#
#
# left_colors = hr["left"].map(lambda s : "g"  if s==0 else "r")
# df_pca.plot(x=0,y=1,kind='scatter', c = left_colors)
#
# # Non-linear dimensionality reduction through Isometric Mapping
# iso = Isomap(n_components=2, n_neighbors=40)
# iso_representation = iso.fit_transform(hr_norm)
# df_iso = pd.DataFrame(iso_representation)
# df_iso.head()
#
#
# df_iso.plot(x=0,y=1,kind='scatter', c = left_colors)
#
#
#
#
# hr_stay = hr[hr["left"]==0]
# hr_left = hr[hr["left"]==1]
#
# fig, axes = plt.subplots(ncols=3,figsize=(10,6))
# sns.factorplot(y="satisfaction_level",x="left",data=hr,kind="box", ax=axes[0])
# axes[1].hist(hr_stay["satisfaction_level"],bins=50,label="Stay",alpha=0.7)
# axes[1].hist(hr_left["satisfaction_level"],bins=50,label="Left",alpha=0.7)
# axes[1].set_xlabel("Satifaction level")
# axes[1].set_ylabel("Count")
# axes[1].legend()
#
#
# g = sns.kdeplot(data=hr_stay["satisfaction_level"],color='b',shade=True,ax=axes[2])
# g = sns.kdeplot(data=hr_left["satisfaction_level"],color='g',shade=True, ax=axes[2])
# g.legend(["Stay","Left"])
# g.set_xlabel('Satifsfaction level')
# g.set_ylabel('Density')
#
#
# fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(10,6))
# sns.factorplot(y="last_evaluation",x="left",data=hr,kind="box", ax=axes[0])
# axes[1].hist(hr_stay["last_evaluation"],bins=50,label="Stay",alpha=0.7)
# axes[1].hist(hr_left["last_evaluation"],bins=50,label="Left",alpha=0.7)
# axes[1].set_xlabel("last_evaluation")
# axes[1].set_ylabel("Count")
# axes[1].legend()
#
# g = sns.kdeplot(data=hr_stay["last_evaluation"],color='b',shade=True,ax=axes[2])
# g = sns.kdeplot(data=hr_left["last_evaluation"],color='g',shade=True, ax=axes[2])
# g.legend(["Stay","Left"])
# g.set_xlabel('last_evaluation')
# g.set_ylabel('Density')
#
# salary_counts = hr.groupby(['left'])['salary'].value_counts(normalize=True).\
# rename('percentage').mul(100).reset_index()
#
# fig, axes = plt.subplots(ncols=1,figsize=(12,6))
# g = sns.barplot(x='salary',y='percentage',data=salary_counts,
#             hue='left')
# g.set_ylabel('percentage')
#



hr = pd.read_csv('HR_comma_sep.csv')
sales_counts = hr.groupby(['left'])['sales'].value_counts(normalize=True).\
rename('percentage').mul(100).reset_index()
a = 1


fig, axes = plt.subplots(ncols=1,figsize=(12,8))
g = sns.barplot(x='sales',y='percentage',data=sales_counts,
            hue='left')
plt.setp(g.get_xticklabels(), rotation=45)
g.set_ylabel('percentage')



#reload HR dataset
hr = pd.read_csv('HR_comma_sep.csv')
hr.head()

#convert salary feature to an ordered categorical variable
hr["salary"] = hr["salary"].\
                    astype("category",ordered=True, \
                    categories = ['low','medium','high']).cat.codes
# pairplot uses scatterplots and histograms by default
# g = sns.pairplot(hr.drop(labels=['salary','sales','number_project'\
#                ,'time_spend_company','Work_accident',
#                'promotion_last_5years'],axis=1),hue='left')
# g.map_diag(plt.hist)
# g.map_offdiag(plt.scatter)
# g.add_legend()


hr = pd.read_csv('HR_comma_sep.csv')

hr.info()

x_train = pd.get_dummies(hr.drop(labels='left',axis=1))
y_train = hr['left']

X = x_train[['satisfaction_level']]

#import estimator
from sklearn.linear_model import LogisticRegression
#instantiate estimator to crate an estimator object
lr = LogisticRegression()
type(lr)


# use k-fold cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True, random_state = 1234)
new_scores = cross_val_score(lr,X,y_train,cv=kf)
print(new_scores)
print(new_scores.mean())

# use stratified k-fold cross validation
from sklearn.model_selection import cross_val_score
new_scores = cross_val_score(lr,x_train,y_train,cv=5)
print(new_scores)

from sklearn.tree import DecisionTreeClassifier
#instantiate estimator to crate an estimator object
DTC = DecisionTreeClassifier(max_depth=3)

DTC.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
kf = KFold(n_splits = 5, shuffle = True, random_state = 1234)
new_scores = cross_val_score(DTC,x_train,y_train,cv=kf)
print(new_scores)
print(new_scores.mean())


#import estimator
from sklearn.ensemble import RandomForestClassifier
#instantiate estimator to crate an estimator object
rfc = RandomForestClassifier(n_estimators=100)
type(rfc)

# use stratified k-fold cross validation
new_scores = cross_val_score(rfc,x_train,y_train,cv=5)
print(new_scores)
print(new_scores.mean())

rfc.fit(x_train,y_train)
rfc.feature_importances_

feature_names = x_train.columns

importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(rfc.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=True).set_index('feature')
importances.plot.barh()
