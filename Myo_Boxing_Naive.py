
# coding: utf-8

# In[36]:

# imports
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[37]:

dir = './'

X = np.empty([101, 910])
# X = np.empty(shape=(0,0))
for root,dirs,files in os.walk(dir):
    for idx,file in enumerate(files):
        if file.endswith(".csv") and file != "output.csv":
            # print (file)
            curr_file = pd.read_csv(file, header=None)
            headers = curr_file.iloc[0, :]
            curr_file = curr_file.iloc[2:, 1:]
            new_punch = np.matrix.flatten(curr_file.values)
            new_punch = new_punch[0:910]
            X[idx-1] = new_punch
        
        if file == "output.csv":
            # print ("found em")
            output_file = pd.read_csv(file, header=None)
            headers = output_file.iloc[0, :]
            output_file = output_file.iloc[2:, 1:]
            new_test_punch = np.matrix.flatten(output_file.values)
            new_test_punch = new_test_punch[0:910]


# In[38]:

# create X and y
# X = np.split(punches, [11, 910])
# y = np.split(punches, :, 911)
# X = np.array(X)
y0 = np.zeros(51)
y1 = np.ones(50)
y = np.append(y0,y1)
# y = np.ndarray.tolist(y)

y_reshape = np.reshape(y, (-1,1))
print (X.shape, y.shape, y_reshape.shape)
train_data = np.concatenate((X,y_reshape), axis=1)


# In[39]:

# train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print (X_train.shape)
# print (X_test)
# print (y_test)


# ### Naive Bayes

# In[40]:

# # bayesian classifier
# gnb = GaussianNB()
# model = gnb.fit(X, y)

# y_pred = gnb.predict(new_test_punch)
# y_pred_proba = gnb.predict_proba(new_test_punch)

# # xgboost
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2

# train_data = train_data.astype(float)
# new_test_punch = new_test_punch.astype(float)

# train_data = xgb.DMatrix(train_data, label=y)
# new_test_punch = xgb.DMatrix(new_test_punch)

# bst = xgb.train(param, train_data, num_round)
# y_pred = bst.predict(new_test_punch)

# SVC
clf = SVC(kernel="poly", C=1, probability=True)
clf.fit(X, y)

score = clf.predict_proba(new_test_punch)
print (score)

# print(y_pred)
# if y_pred == 1:
#     print ("Damn rockay!!!")
    # print ("Your probability is ", y_pred)
# else:
#     print ("ur accuracy is p00p")
    # print ("Your probability is ", y_pred)

# print ('y test: ', y_test)
# print ('y pred: ', y_pred)

# print results


