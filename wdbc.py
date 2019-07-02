# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:38:19 2019

@author: DEEPSHIKHA
"""


import pandas as pd
import numpy as np 

cncr = pd.read_csv('wdbc.csv')

columns = ['id', 'diagnosis', 'rad_mean', 'text_mean', 'peri_mean', 'area_mean', 'smooth_mean', 'comp_mean', 'conc_mean','concv_pt_mean', 'sym_mean', 'fract_dim_mean', 'rad_se', 'text_se' , 'per_se', 'area_se', 'smooth_se', 'comp_se','concv_se', 'concv_pt_se','sym_se', 'fract_dim_se' , 'rad_wor','text_wor','per_wor', 'area_wor', 'smooth_wor', 'comp_wor', 'concv_wor','concv_pt_wor','sym_wor', 'frac_dim_wor']

cncr.columns = columns

############# Standardize the data ############################
cncr
    
X = cncr.iloc[:, 2:].values
Y = cncr.iloc[:, 1].values

X
Y


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

######################### MAPPING THE VALUES ######################################
cncr['diagnosis'] = cncr['diagnosis'].map({'M':1 , 'B' :2})

################## MACHINE LEARNING ALGORITHMS ###########################
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/5, random_state = 5)

############# svM ###################
from sklearn.svm import SVC
classifier = SVC(gamma = 'auto')
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_pred, Y_test)
print(acc)


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_pred, Y_test)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print(TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP) 
print(TNR)


######################

precision = TP / (TP + FP)
print(precision)

recall = TP / (TP + FN)
print(recall)

f1 = 2 * precision * recall / (precision + recall)
print(f1)



######################## linear regression as a classifier ########################3

#from sklearn.linear_model import LinearRegression
#classifier = LinearRegression()


###################### sgd ###############################
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_pred, Y_test)
print(acc1)

############# RANDOM FOREEST CLASSIFIER############

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc2 = accuracy_score(Y_pred, Y_test)
print(acc2)


from sklearn.metrics import confusion_matrix 
cm1 = confusion_matrix(Y_pred, Y_test)
print(cm1)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print(TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP) 
print(TNR)

###################################

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f1)




################################ GRADIENTBOOSTINGCLASSIFIER##################

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(n_estimators=100, random_state=5)
GBC.fit(X_train, Y_train)
Y_pred = GBC.predict(X_test)


from sklearn.metrics import accuracy_score
acc3 = accuracy_score(Y_pred, Y_test)
print(acc3)

################################ KNN #######################

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5, algorithm = 'kd_tree')
neigh.fit(X_train, Y_train) 

Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
acc2 = accuracy_score(Y_pred, Y_test)
print(acc2)


x
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print(TNR)
 #####################################
 
 
 
 
 
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f1)


################################### Softmax Regression  #################################3
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', penalty = 'l2')
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score
acc5 = accuracy_score(Y_pred, Y_test)
print(acc5)


############################ Multilayer perceptron #########################3
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=50, alpha=0.0001,solver='sgd', verbose=10,  random_state=21)
clf.fit(X_train , Y_train)

Y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
acc6 = accuracy_score(Y_pred, Y_test)
print(acc6)


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_pred, Y_test)
print(cm)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print(TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP) 
print(TNR)

###################################
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f1)







################## Naive bayes #########################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score
acc7 = accuracy_score(Y_pred, Y_test)
print(acc7)


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(Y_pred, Y_test)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]


# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print(TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP) 
print(TNR)
#######################
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(f1)





############################ MULTINOMIAL NB ########################################
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score
acc8 = accuracy_score(Y_pred, Y_test)
print(acc8)



################################# visualization ####################################

import matplotlib.pyplot as plt
import seaborn as sns 

plt.scatter('rad_mean', 'peri_mean')
plt.title('Radius Vs Perimeter of Nucleus')
plt.xlabel('Radius of nucleus')
plt.ylabel('Perimeter of nucleus')
plt.show()

plt.boxplot(cncr['rad_mean'])
plt.show()


plt.boxplot(cncr['text_mean'])
plt.show()

plt.boxplot(cncr['peri_mean'])
plt.show()

plt.boxplot(cncr['area_mean'])
plt.show()

# plot between 2 attributes 
plt.bar(cncr['comp_mean'], cncr['conc_mean']) 
plt.xlabel("Compactness ") 
plt.ylabel("Concavity") 
plt.show() 

colors = ['#E69F00', '#56B4E9']

plt.hist(column = ['rad_se','text_se'], bins = int(180/15), stacked=True,normed=True, color = colors)

plt.show()


sns.distplot(cncr['diagnosis'], hist=True, kde=True, bins=int(180/5), color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    
sns.set(style="whitegrid")

import seaborn as sns
sns.set(style="whitegrid")
ax = sns.swarmplot(x=cncr['sym_mean'])
ax = sns.swarmplot(x=cncr['rad_mean'])
ax = sns.swarmplot(x=cncr['text_mean'])
ax = sns.swarmplot(x=cncr['peri_mean'])
ax = sns.swarmplot(x=cncr['area_mean'])
ax = sns.swarmplot(x=cncr['smooth_mean'])
ax = sns.swarmplot(x=cncr['comp_mean'])
ax = sns.swarmplot(x=cncr['conc_mean'])
ax = sns.swarmplot(x=cncr['concv_pt_mean'])
ax = sns.swarmplot(x=cncr['fract_dim_mean'])

ax = sns.swarmplot(y=cncr['comp_mean'])


ax = sns.swarmplot(x=cncr['rad_se'])
ax = sns.swarmplot(x=cncr['text_se'])
ax = sns.swarmplot(x=cncr['per_se'])
ax = sns.swarmplot(x=cncr['area_se'])
ax = sns.swarmplot(x=cncr['smooth_se'])
ax = sns.swarmplot(x=cncr['comp_se'])
ax = sns.swarmplot(x=cncr['concv_se'])
ax = sns.swarmplot(x=cncr['concv_pt_se'])
ax = sns.swarmplot(x=cncr['sym_se'])
ax = sns.swarmplot(x=cncr['fract_dim_se'])

ax = sns.swarmplot(x=cncr['rad_wor'])
ax = sns.swarmplot(x=cncr['text_wor'])
ax = sns.swarmplot(x=cncr['per_wor'])
ax = sns.swarmplot(x=cncr['area_wor'])
ax = sns.swarmplot(x=cncr['smooth_wor'])
ax = sns.swarmplot(x=cncr['comp_wor'])
ax = sns.swarmplot(x=cncr['concv_wor'])
ax = sns.swarmplot(x=cncr['concv_pt_wor'])
ax = sns.swarmplot(x=cncr['sym_wor'])
ax = sns.swarmplot(x=cncr['frac_dim_wor'])

correlation = cncr.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, linewidths = 0.5, vmin=-1, cmap="PuBu_r")


