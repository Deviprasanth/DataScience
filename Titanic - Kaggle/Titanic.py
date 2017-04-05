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
print(test_df.head(15))

