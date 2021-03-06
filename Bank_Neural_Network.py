#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import pydotplus
from IPython.display import Image, display
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
dot_data = StringIO()

bank = pd.read_csv('Shubham-data/bank.csv')
bank.head()

# Check if the data set contains any null values - Nothing found!
bank[bank.isnull().any(axis=1)].count()

g = sns.boxplot(x=bank["age"])

bank_data = bank.copy()

# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data['poutcome'] = bank_data['poutcome'].replace(['other'] , 'unknown')

# Make a copy for parsing
bank_data.poutcome.value_counts()

# Combine similar jobs into categories
bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
bank_data['job'] = bank_data['job'].replace(['services','housemaid'], 'pink-collar')
bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')

bank_data.job.value_counts()
# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data['poutcome'] = bank_data['poutcome'].replace(['other'], 'unknown')
bank_data.poutcome.value_counts()

# Drop 'contact', as every participant has been contacted.
bank_data.drop('contact', axis=1, inplace=True)

# values for "default" : yes/no
bank_data['default_cat'] = bank_data['default'].map({'yes': 1, 'no': 0})
bank_data.drop('default', axis=1, inplace=True)

# values for "housing" : yes/no
bank_data["housing_cat"]=bank_data['housing'].map({'yes':1, 'no':0})
bank_data.drop('housing', axis=1,inplace = True)

# values for "loan" : yes/no
bank_data["loan_cat"] = bank_data['loan'].map({'yes':1, 'no':0})
bank_data.drop('loan', axis=1, inplace=True)

# day  : last contact day of the month
# month: last contact month of year
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)

# values for "deposit" : yes/no
bank_data["deposit_cat"] = bank_data['deposit'].map({'yes':1, 'no':0})
bank_data.drop('deposit', axis=1, inplace=True)

# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000

# Create a new column: recent_pdays
bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data.pdays, 1/bank_data.pdays)

# Convert categorical variables to dummies
bank_with_dummies = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'],                                    prefix = ['job', 'marital', 'education', 'poutcome'])
bank_with_dummies.head()

bank_with_dummies.describe()

bank_with_dummies.plot(kind='scatter', x='age', y='balance');

bankcl = bank_with_dummies
corr = bankcl.corr()

bank_with_dummies[bank_data.deposit_cat == 1].describe()

data_drop_deposite = bankcl.drop('deposit_cat', 1)
label = bankcl.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)


# In[39]:


get_ipython().run_cell_magic('time', '', 'clf = MLPClassifier(activation="relu", solver=\'lbfgs\', alpha=1e-5, hidden_layer_sizes=(6), random_state=1, max_iter=800)\nprint(clf.fit(data_train, label_train))\nprint(clf.score(data_train, label_train))')


# In[37]:


get_ipython().run_cell_magic('time', '', "parameters = {'solver': ['lbfgs', 'sgd'], 'alpha':10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0,2,4,6,8]}\nclf = GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=5)\nclf.fit(data_train, label_train)\nclf_model = clf.best_estimator_\nprint (clf.best_score_, clf.best_params_) ")


# In[40]:


get_ipython().run_cell_magic('time', '', 'f1_score(label_test, clf.predict(data_test), average=\'macro\')  \nprint(confusion_matrix(label_test, clf.predict(data_test)))\nprint(classification_report(label_test, clf.predict(data_test))) \n# clf_score_train = clf.score(data_train, label_train)\n# print("Training score: ",clf_score_train)\n# clf_score_test = clf.score(data_test, label_test)\n# print("Testing score: ",clf_score_test)')


# In[29]:


param_range = np.arange(1, 10)
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='lbfgs', alpha=1e-07, random_state=4), data_train, label_train, param_name="hidden_layer_sizes", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Artificial Neural Network on the Bank Data")
plt.xlabel("Hidden Layer Sizes")
#plt.xticks(param_range)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean,'o-', label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="r", lw=lw)
plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[35]:


param_range = 10.0 ** -np.arange(-10, 20)
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='lbfgs', alpha=1e-07, hidden_layer_sizes=6, random_state=4), data_train, label_train, param_name="alpha", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Artificial Neural Network on the Bank Data")
plt.xlabel("alpha")
#plt.xticks(param_range)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean,'o-', label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="r", lw=lw)
plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[36]:


param_range = np.arange(1, 2000, 200)
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='lbfgs', hidden_layer_sizes=6, random_state=4), data_train, label_train, param_name="max_iter", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Artificial Neural Network on the Bank Data")
plt.xlabel("max_iter")
#plt.xticks(param_range)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean,'o-', label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="r", lw=lw)
plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[7]:


plt.figure()
plt.title("Learning curve with Artificial Neural Network on the Bank Data")
plt.xlabel("Training examples")
plt.ylabel("Score")

train_sizes = np.arange(1, 7000, 500)
train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(solver='lbfgs', alpha=1e-07, hidden_layer_sizes=(6), random_state=4), data_train, label_train, cv=5, n_jobs=4, train_sizes=train_sizes)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std  = np.std(train_scores, axis=1)
test_scores_mean  = np.mean(test_scores, axis=1)
test_scores_std   = np.std(test_scores, axis=1)
plt.grid()
lw = 2
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,
                 color="r", lw=lw)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

plt.legend(loc="best")


# In[ ]:




