#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pathlib
import imageio
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt        # to plot any graph
import cv2
import os
import random
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import imageio
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt        # to plot any graph
import cv2
import os
import random
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn import svm
from sklearn import preprocessing

flowers = glob(os.path.join('Shubham-data/17flowers/jpg', "*.jpg"))
aeroplanes = glob(os.path.join('Shubham-data/airplanes_side', "*.jpg"))

# flowers_sorted = sorted([x for x in flowers])
# print(flowers)
# im_path = flowers_sorted[45]
# im = imageio.imread(str(im_path))
# print(im.shape)
# im_gray = rgb2gray(im)
# im_gray.reshape(-1,3)
# print('New image shape: {}'.format(im_gray.shape))

# plt.imshow(im, cmap='Set3')  # show me the leaf
# plt.show()

# plt.imshow(im_gray, cmap='Set3')  # show me the leaf
# plt.show()

def proc_images(images, label):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """



    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 64
    HEIGHT = 64

    for img in images:

        # Read and resize image
        full_size_image = cv2.imread(img, 0)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))

        # Labels
        y.append(label)



    return x,y



flowers_x, flowers_y = proc_images(flowers, 1)
aeroplanes_x, aeroplanes_y = proc_images(aeroplanes, 0)
len(flowers_x)

df = pd.DataFrame()
df["labels"]=aeroplanes_y, flowers_y
df["images"]=aeroplanes_x, flowers_x
df.head()



plt.imshow(flowers_x[0], cmap='Set3')

plt.imshow(aeroplanes_x[0], cmap='Set3')



for i in range(len(flowers_x)):
    flowers_x[i] = flowers_x[i].flatten()

for i in range(len(aeroplanes_x)):
    aeroplanes_x[i] = aeroplanes_x[i].flatten()


training = flowers_x + aeroplanes_x
labels = flowers_y + aeroplanes_y
data_train, data_test, label_train, label_test = train_test_split(training, labels, test_size = 0.2, random_state = 50)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(data_train)
X_test_minmax = min_max_scaler.fit_transform(data_test)
# print(data_train)



# dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)
# dt2.fit(training, labels)
# dt2_score_train = dt2.score(data_train, label_train)
# print("Training score: ",dt2_score_train)
# dt2_score_test = dt2.score(data_test, label_test)
# print("Testing score: ",dt2_score_test)


# # Let's generate the decision tree for depth = 6
# # Create a feature vector
# #features = training.columns.tolist()

# # Uncomment below to generate the digraph Tree.
# #tree.export_graphviz(dt2, out_file=dot_data, feature_names=features)


# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200), random_state=1)
# clf.fit(data_train, label_train)
# print(clf.score(data_train, label_train))
# print(clf.score(data_test, label_test))
# print(clf.predict(data_test))


# In[17]:


get_ipython().run_cell_magic('time', '', "clf = svm.SVC(kernel='linear')\n# Train Adaboost Classifer\nclf.fit(X_train_minmax, label_train)\n\n \nprint(confusion_matrix(label_test, clf.predict(X_test_minmax)))\nprint(classification_report(label_test, clf.predict(X_test_minmax)))  ")


# In[20]:


print(f1_score(label_test, clf.predict(X_test_minmax), average='macro'))
print(clf)


# In[18]:


get_ipython().run_cell_magic('time', '', "svclassifierrbf = svm.SVC(kernel='rbf', gamma = 0.001)  \nsvclassifierrbf.fit(X_train_minmax, label_train)  \n \nprint(confusion_matrix(label_test, svclassifierrbf.predict(X_test_minmax)))\nprint(classification_report(label_test, svclassifierrbf.predict(X_test_minmax)))  ")


# In[21]:


print(f1_score(label_test, svclassifierrbf.predict(X_test_minmax), average='macro'))


# In[9]:


get_ipython().run_cell_magic('time', '', "# Create adaboost classifer object\nparameters = {'gamma' : [0.001, 0.01, 0.1, 1]}\nclf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=5, n_jobs=-1)\nclf.fit(data_train, label_train)\nprint (clf.best_score_, clf.best_params_) ")


# In[10]:


get_ipython().run_cell_magic('time', '', "# Create Linear SVM classifer object\nparameters = {'C' : [1,5,10]}\nclflinear = GridSearchCV(svm.SVC(kernel='linear'), parameters, cv=5, n_jobs=-1)\nclflinear.fit(data_train, label_train)\nprint (clflinear.best_score_, clflinear.best_params_) ")


# In[22]:


param_range = [10,20,50,100,500,1000,2000,5000,10000]
train_scores, test_scores = validation_curve(
    svm.SVC(kernel='linear'), X_train_minmax, label_train, param_name="max_iter", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Linear KERERL SVM Classifier on the Image Data")
plt.xlabel("max_iter")
#plt.xticks(param_range)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[27]:


plt.figure()
plt.title("Learning curve with Linear KERNEL SVM on the Image Data")
plt.xlabel("Training examples")
plt.ylabel("Score")

train_sizes = np.linspace(.1, 1.0, 20)
train_sizes, train_scores, test_scores = learning_curve(svm.SVC(kernel='linear'), X_train_minmax, label_train, cv=5, n_jobs=-1, train_sizes=train_sizes)

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


# In[25]:


param_range = [0.0005, 0.001, 0.01, 0.1, 1]
train_scores, test_scores = validation_curve(
    svm.SVC(kernel='rbf'), X_train_minmax, label_train, param_name="gamma", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RBF KERERL SVM Classifier on the Image Data")
plt.xlabel("gamma")
#plt.xticks(param_range)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[24]:



print(X_train_minmax)
param_range = [10,20,50,100,500,1000,2000,5000,10000]
train_scores, test_scores = validation_curve(
    svm.SVC(kernel='rbf'), X_train_minmax, label_train, param_name="max_iter", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with RBF KERERL SVM Classifier on the Image Data")
plt.xlabel("max_iter")
#plt.xticks(param_range)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()


# In[26]:


plt.figure()
plt.title("Learning curve with RBF KERNEL SVM on the Image Data")
plt.xlabel("Training examples")
plt.ylabel("Score")

train_sizes = np.linspace(.1, 1.0, 20)
train_sizes, train_scores, test_scores = learning_curve(svm.SVC(kernel='rbf', gamma = 0.001), X_train_minmax, label_train, cv=5, n_jobs=-1, train_sizes=train_sizes)

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




