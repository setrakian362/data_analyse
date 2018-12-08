import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Importing the datasets
train = pd.read_csv("input/train_for_model.csv")
test = pd.read_csv("input/test_for_model.csv")
a= 1

# separating our independent and dependent variable
X = train.drop(['Survived'], axis=1)
y = train["Survived"]

#age_filled_data_nor = NuclearNormMinimization().complete(df1)
#Data_1 = pd.DataFrame(age_filled_data, columns = df1.columns)
#pd.DataFrame(zip(Data["Age"],Data_1["Age"],df["Age"]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .33, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

test = sc.transform(test)


## Necessary modules for creating models.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix


# Logistic Regression¶

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = logreg, X = x_train, y = y_train, cv = 10, n_jobs = -1)
logreg_accy = accuracies.mean()
print (round((logreg_accy),3))

if __name__ == '__main__':
    from sklearn.model_selection import GridSearchCV
    C_vals = [0.099,0.1,0.2,0.5,12,13,14,15,16,16.5,17,17.5,18]
    penalties = ['l1','l2']

    param = {'penalty': penalties,
             'C': C_vals
            }
    grid_search = GridSearchCV(estimator=logreg,
                               param_grid = param,
                               scoring = 'accuracy',
                               cv = 10
                              )