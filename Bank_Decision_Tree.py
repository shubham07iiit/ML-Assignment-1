#!/usr/bin/env python
# coding: utf-8

# In[43]:


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
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
dot_data = StringIO()

bank = pd.read_csv('Shubham-data/bank.csv')
bank.head()

# Check if the data set contains any null values - Nothing found!
bank[bank.isnull().any(axis=1)].count()

g = sns.boxplot(x=bank["age"])

bank_data = bank.copy()
print(bank_data.shape)

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

bank_with_dummies.plot(kind='scatter', x='duration', y='balance');
bank_with_dummies.plot(kind='scatter', x='age', y='balance');

bankcl = bank_with_dummies
corr = bankcl.corr()

bank_with_dummies[bank_data.deposit_cat == 1].describe()

data_drop_deposite = bankcl.drop('deposit_cat', 1)
label = bankcl.deposit_cat
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)
data_train.shape
# data_test =  (30, 1500, 1000, 4, 10000, 0, 0)


# In[40]:


get_ipython().run_cell_magic('time', '', "parameters = {'max_depth':range(1,10)}\nclf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=4, cv=5)\nclf.fit(data_train, label_train)\ntree_model = clf.best_estimator_\nprint (clf.best_score_, clf.best_params_) \n\nf1_score(label_test, clf.predict(data_test), average='macro')  \nprint(label_test.shape)\nprint(confusion_matrix(label_test, clf.predict(data_test)))\n\n# print(clf.predict(data_test))")


# In[22]:


print(classification_report(label_test, clf.predict(data_test)))


# In[16]:


# Decision tree with depth = 6
clf_score_train = clf.score(data_train, label_train)
print("Training score: ",clf_score_train)
clf_score_test = clf.score(data_test, label_test)
print(accuracy_score(label_test, clf.predict(data_test)))
print("Testing score: ",clf_score_test)





# In[25]:


# Uncomment below to generate the digraph Tree.
clf = tree.DecisionTreeClassifier(random_state=1, max_depth=7)
clf.fit(data_train, label_train)
# Let's generate the decision tree for depth = 6
# Create a feature vector
features = data_drop_deposite.columns.tolist()

# Uncomment below to generate the digraph Tree.
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph[0].create_png()))


# In[37]:


with open("bank_desicion_tee.png", "wb") as png:
    png.write((graph[0].create_png()))


# In[45]:


fi = clf.best_estimator_.feature_importances_

l = len(features)
for i in range(0,len(features)):
    print('{:.<20} {:3}'.format(features[i],fi[i]))


# In[17]:


get_ipython().run_cell_magic('time', '', 'param_range = np.arange(1, 10, 1)\ntrain_scores, test_scores = validation_curve(\n    tree.DecisionTreeClassifier(class_weight=\'balanced\'), data_train, label_train, param_name="max_depth", param_range=param_range,\n    cv=5, scoring="accuracy", n_jobs=4)\ntrain_scores_mean = np.mean(train_scores, axis=1)\ntrain_scores_std = np.std(train_scores, axis=1)\ntest_scores_mean = np.mean(test_scores, axis=1)\ntest_scores_std = np.std(test_scores, axis=1)\n\nplt.title("Validation Curve with Decision Tree Classifier on the Bank Data")\nplt.xlabel("max_depth")\n#plt.xticks(param_range)\nplt.ylabel("Score")\nplt.ylim(0.0, 1.1)\nlw = 2\nplt.plot(param_range, train_scores_mean, label="Training score",\n             color="darkorange", lw=lw)\nplt.fill_between(param_range, train_scores_mean - train_scores_std,\n                 train_scores_mean + train_scores_std, alpha=0.2,\n                 color="darkorange", lw=lw)\nplt.plot(param_range, test_scores_mean, label="Cross-validation score",\n             color="navy", lw=lw)\nplt.fill_between(param_range, test_scores_mean - test_scores_std,\n                 test_scores_mean + test_scores_std, alpha=0.2,\n                 color="navy", lw=lw)\nplt.legend(loc="best")\nplt.show()')


# In[19]:


get_ipython().run_cell_magic('time', '', 'plt.figure()\nplt.title("Learning curve with Decision Tree on the Bank Data")\nplt.xlabel("Training examples")\nplt.ylabel("Score")\n\ntrain_sizes = np.arange(1, 7000, 500)\ntrain_sizes, train_scores, test_scores = learning_curve(tree.DecisionTreeClassifier(class_weight=\'balanced\', max_depth=7), data_train, label_train, cv=5, n_jobs=4, train_sizes=train_sizes)\n\ntrain_scores_mean = np.mean(train_scores, axis=1)\ntrain_scores_std  = np.std(train_scores, axis=1)\ntest_scores_mean  = np.mean(test_scores, axis=1)\ntest_scores_std   = np.std(test_scores, axis=1)\nplt.grid()\nlw = 2\nplt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,\n                 color="r", lw=lw)\nplt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2, color="g")\nplt.plot(train_sizes, train_scores_mean, \'o-\', color="r",label="Training score")\nplt.plot(train_sizes, test_scores_mean, \'o-\', color="g",label="Cross-validation score")\n\nplt.legend(loc="best")')


# In[ ]:





# In[ ]:




