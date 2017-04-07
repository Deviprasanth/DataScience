# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

#==========================================================

# Acquiring Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#==========================================================
'''
# Analyze the Data acquired


train_df.info()
print('_'*40)

test_df.info()
print('_'*40)
# Ordinal - Pclass (ordinal - also categorical with ordering i.e. first class better than second and third)
# Categorical - Survived, Sex, Embarked

print(train_df.describe())

print(train_df.describe(include=['O']))
'''
#==========================================================
# Assumptions and Corrections:
'''
1) Correlating

2) Completing
    Age is an important feature in survival and is missing in some entries -
    -this needs to completed
    Embarked is also need to be completed as we do not know its affect yet

3) Correcting
    Ticket feature may be dropped as it may be reflected in class and fare
    Cabin also can be dropped as cabins are duplicated among different members-
    - and also these are reflected in Pclass
    lly name and passengerid are dropped

4) Creating
    May convert age from numerical to ordinal
    May want to convert Fare from numerical to ordinal (or Pclass might be enough)
    If not dropped, then we may use the title from name (Dr. etc. )
    We may also group the members according to family-
    - (Same Family might have same survive status)
    
5) Classifying
    Woman > Men for Surviving
    Pclass = 1 and children have better survival

'''
#==========================================================
#Analyze by grouping features:
'''

# Pclass vs Survived
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Sex vs Survived
print(
train_df[['Sex','Survived']].groupby('Sex',as_index=False).mean()
    )

# Parch vs Survived
print(
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
)
# Sibsp vs Survived
print(
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
)

# Analyze Age vs Survival
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
sns.plt.show()

# Class and Age vs Survival
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
sns.plt.show()

# Embarked, Age and Sex vs Survival
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
sns.plt.show()


'''
#==========================================================
# Data Wrangling:

#Till now the Data is just analysed but is not cropped/dropped or corrected


# Drop Ticket and Cabin
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df] # For implementing at the same time

# Creating new column 'Title'
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

'''
print(pd.crosstab(train_df['Title'], train_df['Sex']))

Sex       female  male
Title                 
Capt           0     1
Col            0     2
Countess       1     0
Don            0     1
Dr             1     6
Jonkheer       0     1
Lady           1     0
Major          0     2
Master         0    40
Miss         182     0
Mlle           2     0
Mme            1     0
Mr             0   517
Mrs          125     0
Ms             1     0
Rev            0     6
Sir            0     1

AS we can see, a lot of titles are too small to consider. Lets combine all the small titles
'''
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Now lets convert the titles to ordinals in ascending order

title_mapping = {"Mr": 1, "Miss": 4, "Mrs": 5, "Master": 3, "Rare": 2}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Now we drop PassengerId and Name
train_df = train_df.drop(['PassengerId','Name'],axis = 1)
test_df = test_df.drop('Name',axis=1)
combine =[train_df,test_df]

# Convert Sex to Ordinal
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# print(train_df.describe())
'''
Till now we did not handle with the missing values. As we saw above that the age is
- important and we need to handle these.

Age:

'''
train_df = train_df.dropna() # Training cannot be done on guessed values

combine =[train_df,test_df]
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            #guess_ages[i,j] = age_guess
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

'''
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'],
                                             as_index=False).mean().sort_values(by='Survived',
                                                                                ascending=False))

'''

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# print(train_df.head(10))

freq_port = train_df.Embarked.dropna().mode()[0]    # Highest freq port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
'''
print(
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    )

  Embarked  Survived
0        C  0.607692
2        S  0.362816
1        Q  0.285714

'''
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 1, 'C': 2, 'Q': 0} ).astype(int)

'''
test_df.describe()
       PassengerId      Pclass         Sex         Age        Fare  \
count   418.000000  418.000000  418.000000  418.000000  417.000000   
mean   1100.500000    2.265550    0.363636    1.643541   35.627188   
std     120.810458    0.841838    0.481622    4.916055   55.907576   
min     892.000000    1.000000    0.000000    0.000000    0.000000   
25%     996.250000    1.000000    0.000000    1.000000    7.895800   
50%    1100.500000    3.000000    0.000000    1.000000   14.454200   
75%    1204.750000    3.000000    1.000000    2.000000   31.500000   
max    1309.000000    3.000000    1.000000   76.000000  512.329200   

         Embarked       Title     IsAlone  
count  418.000000  418.000000  418.000000  
mean     0.598086    2.370813    0.605263  
std      0.854496    1.673266    0.489380  
min      0.000000    1.000000    0.000000  
25%      0.000000    1.000000    0.000000  
50%      0.000000    1.000000    1.000000  
75%      1.000000    4.000000    1.000000  
max      2.000000    5.000000    1.000000 
'''

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
'''
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

         FareBand  Survived
0       [0, 8.05]  0.204188
1  (8.05, 15.646]  0.321212
2    (15.646, 33]  0.486034
3   (33, 512.329]  0.615819
'''
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 8.05, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 15.646), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 15.646) & (dataset['Fare'] <= 33), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 33, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

'''
>>> train_df.head(10)
    Survived  Pclass  Sex  Age  Fare  Embarked  Title  IsAlone
0          0       3    0    1     0         1      1        0
1          1       1    1    2     3         2      5        0
2          1       3    1    1     0         1      4        1
3          1       1    1    2     3         1      5        0
4          0       3    0    2     0         1      1        1
6          0       1    0    3     3         1      1        1
7          0       3    0    0     2         1      3        0
8          1       3    1    1     1         1      5        0
9          1       2    1    0     2         2      5        0
10         1       3    1    0     2         1      4        0
>>> test_df.head(10)
   PassengerId  Pclass  Sex  Age  Fare  Embarked  Title  IsAlone
0          892       3    0    2     0         0      1        1
1          893       3    1    2     0         1      5        0
2          894       2    0    3     1         0      1        1
3          895       3    0    1     1         1      1        1
4          896       3    1    1     1         1      5        0
5          897       3    0    0     1         1      1        1
6          898       3    1    1     0         0      4        1
7          899       2    0    1     2         1      1        0
8          900       3    1    1     0         2      5        1
9          901       3    0    1     2         1      1        0
'''
#==========================================================
#==========================================================
# Model , Predict and Solve
#==========================================================
'''

Model, predict and solve
Now we are ready to train a model and predict the required solution.
There are 60+ predictive modelling algorithms to choose from.
We must understand the type of problem and solution requirement to
narrow down to a select few models which we can evaluate.
Our problem is a classification and regression problem.
We want to identify relationship between output (Survived or not)
with other variables or features (Gender, Age, Port...).
We are also perfoming a category of machine learning which
is called supervised learning as we are training our model with a given dataset.
With these two criteria - Supervised Learning plus Classification and
Regression,we can narrow down our choice of models to a few.
These include:

Logistic Regression
KNN or k-Nearest Neighbors
Support Vector Machines
Naive Bayes classifier
Decision Tree
Random Forrest
Perceptron
Artificial neural network
RVM or Relevance Vector Machine

'''

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

'''
>>> X_train.shape, Y_train.shape, X_test.shape
((712, 7), (712,), (418, 7))
'''
#==========================================================
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
'''
>>> acc_log
80.060000000000002
'''

'''
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

>>> coeff_df.sort_values(by='Correlation', ascending=False)
    Feature  Correlation
5     Title     0.945309
4  Embarked     0.641156
6   IsAlone     0.160236
2       Age    -0.030701
1       Sex    -0.215370
3      Fare    -0.238101
0    Pclass    -1.206483

class has high negative correlation 
'''
#==========================================================
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
'''
>>> acc_svc
83.709999999999994
'''
#==========================================================
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
'''
>>> acc_knn
83.290000000000006
'''
#==========================================================
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
'''
>>> acc_gaussian
78.230000000000004
'''
#==========================================================
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

#==========================================================
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

#==========================================================
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
#==========================================================
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
#==========================================================
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#==========================================================
# Gradient Boosting
gboost = GradientBoostingClassifier( learning_rate=0.1,n_estimators=100)
gboost.fit(X_train, Y_train)
Y_pred = gboost.predict(X_test)
gboost.score(X_train, Y_train)
acc_gboost = round(gboost.score(X_train, Y_train) * 100, 2)

#==========================================================
#==========================================================
### Model evaluation
#==========================================================
'''
We can now rank our evaluation of all the models to choose
the best one for our problem. While both Decision Tree
and Random Forest score the same,we choose to use Random Forest
as they correct for decision trees' habit of
overfitting to their training set. 
'''
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree','Gradient Boosting'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree,acc_gboost]})

'''
>>> models.sort_values(by='Score', ascending=False)
                        Model  Score
3               Random Forest  87.50
8               Decision Tree  87.50
9           Gradient Boosting  84.83
0     Support Vector Machines  83.71
1                         KNN  83.29
7                  Linear SVC  81.04
2         Logistic Regression  80.06
4                 Naive Bayes  78.23
6  Stochastic Gradient Decent  76.83
5                  Perceptron  73.46
'''

#==========================================================
# Exporting the results
#==========================================================
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred # the last run is random forest
    })

submission.to_csv('.tempsubmission3.csv', index=False)
