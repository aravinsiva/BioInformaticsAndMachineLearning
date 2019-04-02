#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import sklearn
import pandas


# In[3]:


import numpy as np
import pandas as pd

 #import molecular biology dataset
url='https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'
names= ['Class','id','Sequence']
data= pd.read_csv(url,names=names)
print (data)


# In[5]:


#build dataset using custom pandas dataset
#each column in a dataframe is a series 

classes= data.loc[:,'Class']
print(classes[:5])


# In[10]:


#divide sequence into individual nucleotides
sequences= list(data.loc[:,'Sequence'])
dataset={}

#loop through sequences and split
for i, seq in enumerate (sequences):
    nucleotides= list(seq)
    nucleotides= [x for x in nucleotides if x!='\t']
    
    #append class assignment
    nucleotides.append(classes[i])
    
    #add to dataset
    dataset[i]=nucleotides

#print first instance to ensure data looks correct
print dataset[0]
    
    


# In[11]:


#turn data back into a pandas datafram
dframe= pd.DataFrame(dataset)
print(dframe)


# In[12]:


#transpose the rows and columns
df= dframe.transpose()
print(df.iloc[:5])


# In[13]:


#rename last column to class
df.rename(columns={57: 'Class'}, inplace=True)
print(df.iloc[:5])


# In[14]:


#Describe the dataset
df.describe()


# In[17]:


#we have an issue as the nucleotides are not a numerical input
series=[]
for name in df.columns:
    series.append(df[name].value_counts())
    
info= pd.DataFrame(series)
details= info.transpose()
print(details)


# In[19]:


#above is the count of each kind of nucleotide
#switch to numerical data
numerical_df= pd.get_dummies(df)
print(numerical_df.iloc[:5])


# In[22]:


#remove one class columns
df= numerical_df.drop(columns=['Class_-'])

df.rename(columns={'Class_+': 'Class'}, inplace= True)
print(df.iloc[:5])


# In[34]:


#Data has been sucessfully preprocessed
#Begin the machine learning 
#Import the algorithms
#Compare n different classification algorithms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

#Compare all of the above clasification algorithms to find the most
#optimal algorithm for this particular dataset


# In[36]:


from sklearn import model_selection
 
X = np.array(df.drop(['Class'],1))
y= np.array(df['Class'])

seed=1

#split into training data and testing data
X_train, X_test, y_train, y_test= model_selection.train_test_split(X,y,test_size=0.25,random_state=seed)


# In[42]:


#define scoring method
scoring= 'accuracy'

#define models to train

names= ["KNN", "Gaussian Process", "Decision Tree", "Random Forest"
       "Neural Network", "AdaBoost", "Naive Bayes", "SVM Linear",
       "SVM RBF", "SVM Sigmoid"]

classifiers= [
    KNeighborsClassifier(n_neighbors=3),
    GaussianProcessClassifier(1.0*RBF(1.0)),
    DecisionTreeClassifier(max_depth= 5),
    RandomForestClassifier(max_depth=5 , n_estimators= 10, max_features= 1),
    MLPClassifier(alpha=1),
    GaussianNB(),
    SVC(kernel='linear'),
    SVC(kernel= 'rbf'),
    SVC(kernel= 'sigmoid')
]

models = zip(names, classifiers)

#evaluate each model in turn

results= []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits= 10, random_state= seed)
    cv_results= model_selection.cross_val_score(model,X_train, y_train,
                                               cv=kfold, scoring=scoring)
    names.append(name)
    msg="{0}: {1}({2})".format(name, cv_results.mean(), cv_results.std())
    print (msg)



# In[43]:


#As seen above the Gaussian Process is the best algorithm based on the
#training data
#Make predictions on the validation dataset

for name, model in models:
    model.fit(X_train, y_train)
    predictions= model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))




# In[ ]:


#As seen the Support Vector Machine with a linear kernel performed the best
#It has been probven that in the field ofk bioinfromatics
#supprt vector machines are among the most commonly used algorithms
#Thus these findings are in line with what is found in industry


