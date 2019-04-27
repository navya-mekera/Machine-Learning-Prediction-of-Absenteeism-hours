#!/usr/bin/env python
# coding: utf-8

# # ****************** MACHINE LEARNING PROJECT *************************
# # TO PREDICT THE ABSENTEEISM IN HOURS OF AN EMPLOYEE

# ## LIBRARIES

# In[11]:


import pandas as pd
import numpy as np
import random
import pylab as pl

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# scikit learning package
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,mean_absolute_error,mean_squared_error,r2_score, make_scorer,classification_report,roc_curve, auc
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler,OneHotEncoder,scale
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from scipy.stats.mstats import mquantiles
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")


# In[12]:


# read csv file
path = 'Absenteeism_at_work_train.csv'
data_530 = pd.read_csv(path, decimal=",")


# ## DATA CLEANUP

# In[13]:


# investigate the data
data_530.info()


# In[14]:


data_530.head(5)


# In[15]:


# age column have junk values

data_530.Age = data_530.Age.replace("R", 0)

data_530['Age'] =data_530['Age'].astype(str).astype('int64')

# replacing rows with age 0
#data_530.Age = data_530.Age.replace(0, int(data_530['Age'].mean()))

data_530 = data_530[data_530['Age'] != 0]


# In[16]:


# Null values in dataset
data_530.isnull().sum()


# In[17]:


# replace null values with respective column mean

data_530 =data_530.fillna(0)


# In[18]:


#type conversion
data_530['Age'] =data_530['Age'].astype('int64')

data_530.describe()


# In[19]:


# cross verify for null values
data_530.isnull().sum()


# In[20]:


# adding new column named 'followUp_req' based on whether reason for absence required follow up or not
data_530['followUp_req'] = np.where(data_530['Reason for absence'] <= 21, 1, 0)

# add categorical target column as per project requirement

data_530['Absenteeism categories'] = np.where(data_530['Absenteeism time in hours'] == 0, "Group 0", 
                                              np.where(data_530['Absenteeism time in hours'] == 1, "Group 1",
                                                      np.where(data_530['Absenteeism time in hours'] == 2, "Group 2",
                                                              np.where(data_530['Absenteeism time in hours'] == 3, "Group 3",
                                                                      np.where((data_530['Absenteeism time in hours'] >= 4)&(data_530['Absenteeism time in hours'] <= 7), "Group 4",
                                                                               np.where(data_530['Absenteeism time in hours'] == 8, "Group 5",
                                                                                       np.where(data_530['Absenteeism time in hours'] >= 9, "Group 6",0))
                                                                              )))))


# In[21]:


# checking for Absenteeism categorie groups 
a= data_530['Absenteeism categories'].unique()
print(sorted(a))


# In[22]:


# formatting to proper data type
data_530['Reason for absence'] = data_530['Reason for absence'].astype('category')
data_530['Month of absence'] = data_530['Month of absence'].astype('category')
data_530['Day of the week'] = data_530['Day of the week'].astype('category')
data_530['Seasons'] = data_530['Seasons'].astype('category')
data_530['Disciplinary failure'] = data_530['Disciplinary failure'].astype('category')
data_530['Education'] = data_530['Education'].astype('category')
data_530['Social drinker'] = data_530['Social drinker'].astype('category')
data_530['Social smoker'] = data_530['Social smoker'].astype('category')
data_530['Pet'] = data_530['Pet'].astype('category')
data_530['followUp_req'] = data_530['followUp_req'].astype('category')
data_530['Absenteeism categories'] = data_530['Absenteeism categories'].astype('category')
data_530.info()


# In[23]:


#observe outliers in transportation expense
print('We can observe outliers in transportation expense')
sns.boxplot(data_530['Transportation expense'])


# In[24]:


sns.boxplot(data_530['Distance from Residence to Work'])
print('no outliers')


# In[25]:


sns.boxplot(data_530['Service time'])
print('We can observe outliers in service time variable')


# In[26]:


sns.boxplot(data_530['Age'])


# In[27]:


sns.boxplot(data_530['Work load Average/day '])

print('We can observe outliers in Work load Average/day')


# In[28]:


sns.boxplot(data_530['Hit target'])
print('We can observe outlier in Hit target')


# In[29]:


sns.boxplot(data_530['Weight'])


# In[30]:


sns.boxplot(data_530['Height'])
print('We can observe outliers in Height variable')


# In[31]:


sns.boxplot(data_530['Body mass index'])


# In[33]:


# store two datasets, one for continous and other categorical
dataset_continuous = data_530.drop('Absenteeism categories', axis=1)
dataset_categorical = data_530.drop('Absenteeism time in hours',axis=1)

print(dataset_continuous.shape)
print(dataset_categorical.shape)


# In[34]:


# write the taining data to file

dataset_continuous.to_csv('cleanDataset_continuousTarget.csv',index=False)
dataset_categorical.to_csv('cleanDataset_categoricalTarget.csv',index=False)


# In[35]:


# get the test dataset
test_path = 'Absenteeism_at_work_test.csv'
mydata_test = pd.read_csv(test_path, decimal=",")


# In[36]:


# preprocess the test dataset
# adding new column named 'followUp_req' based on whether reason for absence required follow up or not
mydata_test['followUp_req'] = np.where(mydata_test['Reason for absence'] <= 21, 1, 0)

# add categorical target column as per project requirement

mydata_test['Absenteeism categories'] = np.where(mydata_test['Absenteeism time in hours'] == 0, "Group 0", 
                                              np.where(mydata_test['Absenteeism time in hours'] == 1, "Group 1",
                                                      np.where(mydata_test['Absenteeism time in hours'] == 2, "Group 2",
                                                              np.where(mydata_test['Absenteeism time in hours'] == 3, "Group 3",
                                                                      np.where((mydata_test['Absenteeism time in hours'] >= 4)&(mydata_test['Absenteeism time in hours'] <= 7), "Group 4",
                                                                               np.where(mydata_test['Absenteeism time in hours'] == 8, "Group 5",
                                                                                       np.where(mydata_test['Absenteeism time in hours'] >= 9, "Group 6",0))
                                                                              )))))

mydata_test['Reason for absence'] = mydata_test['Reason for absence'].astype('category').cat.codes
mydata_test['Month of absence'] = mydata_test['Month of absence'].astype('category').cat.codes
mydata_test['Day of the week'] = mydata_test['Day of the week'].astype('category').cat.codes
mydata_test['Seasons'] = mydata_test['Seasons'].astype('category').cat.codes
mydata_test['Disciplinary failure'] = mydata_test['Disciplinary failure'].astype('category').cat.codes
mydata_test['Education'] = mydata_test['Education'].astype('category').cat.codes
mydata_test['Social drinker'] = mydata_test['Social drinker'].astype('category').cat.codes
mydata_test['Social smoker'] = mydata_test['Social smoker'].astype('category').cat.codes
mydata_test['Pet'] = mydata_test['Pet'].astype('category').cat.codes
mydata_test['followUp_req'] = mydata_test['followUp_req'].astype('category').cat.codes
mydata_test['Absenteeism categories'] = mydata_test['Absenteeism categories'].astype('category').cat.codes


# # EXPLORATORY DATA ANALYSIS

# In[37]:


# load the training dataset
# categorical dataset
mydata_cat = pd.read_csv('cleanDataset_categoricalTarget.csv')
#continous dataset
mydata_con = pd.read_csv('cleanDataset_continuousTarget.csv')


# In[27]:


# Distribution of Reason of Absence
plt.figure(figsize=(10,5))
sns.distplot(mydata_con['Reason for absence'],bins = 28)
plt.title("Reason of Absenece")


# In[28]:


# Absenteeism categories w.r.t seasons
plt.figure(figsize = (10,5))
#fig, axes = plt.subplots(2,2)
sns.catplot(x ='Absenteeism categories',kind = 'count',col = "Seasons",data = mydata_cat)

season_abs_groups = mydata_con.groupby(['Seasons'],as_index = False).agg({'Absenteeism time in hours': "sum"})
print(season_abs_groups)


# In[29]:


# scatter plot of Absenteeism w.r.t Age
with sns.axes_style("white"):
    j = sns.jointplot('Age','Absenteeism time in hours',mydata_con,kind = 'hex',color = "maroon")
    j.annotate(stats.pearsonr)


# In[30]:


# study of age and seasons on Absenteeism
plt.figure(figsize = (20,5))
sns.lmplot(x = 'Age',y = 'Absenteeism time in hours',data = mydata_con,hue = 'Seasons',size=5,aspect=2,legend=False)
plt.legend(['Summer','Winter','Spring','Fall'])


# In[31]:


# line graph for  mean of Absenteeism in hours in different months
plt.figure(figsize=(10,5))
mean_abs_per_month = mydata_con.groupby(['Month of absence','followUp_req'],as_index = False).agg({'Absenteeism time in hours': "mean"})
#print(mean_abs_per_month)
sns.lineplot('Month of absence','Absenteeism time in hours',hue = 'followUp_req',style = 'followUp_req',data = mean_abs_per_month)

plt.legend(['NoFollowup','Followup'])
plt.title("Mean absentism in different months")


# In[32]:


# absenteeism hours
mydata_con.groupby(['Reason for absence']).agg({"Absenteeism time in hours":"sum"})


# In[33]:


# boxplot for weekdays
sns.set_style(style='dark')
j = sns.factorplot("Day of the week", "Absenteeism time in hours", data=mydata_con, kind="box")
labels = ['Monday','Tuesday','Wednesday','Thursday','Friday']
j.set_xticklabels(labels,rotation = 45)
plt.title('Absenteeism on each workingday')


# In[34]:


#Absenteeism hours for different reasons
plt.figure(figsize=(10,8))
sns.set_style(style = "darkgrid")
count_absents = pd.DataFrame(mydata_con.groupby('Reason for absence',as_index = False).agg({'Absenteeism time in hours':'sum'}))
count_absents.sort_values('Absenteeism time in hours',ascending = True,inplace = True)
#print(count_absents)

#plot  the graph
sns.barplot('Reason for absence','Absenteeism time in hours',data = count_absents, color = 'lightblue',edgecolor='black')
plt.title("Absenteeism time for various reasons")


# In[35]:


# correlationship of Distance from work

with sns.axes_style("white"):
    j = sns.jointplot('Distance from Residence to Work','Absenteeism time in hours',mydata_con,kind = 'reg',color = "green")
    j.annotate(stats.pearsonr)


# In[36]:


# correlation w.r.t Transportation expense
with sns.axes_style("whitegrid"):
    j = sns.jointplot('Transportation expense','Absenteeism time in hours',mydata_con,kind = 'reg')
    j.annotate(stats.pearsonr)


# In[37]:


# pair wise grid
grid = sns.PairGrid(data= mydata_con,
                    vars = ['ID','Age', 'Transportation expense', 'Distance from Residence to Work','Absenteeism time in hours'])

# Map a scatter plot to the upper triangle
grid = grid.map_upper(sns.regplot, color = 'blue')


# Map a histogram to the diagonal
grid = grid.map_diag(plt.hist, bins = 10, color = 'red', 
                     edgecolor = 'k')

# Map a density plot to the lower triangle
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')


# ## EXPLORATORY ANALYSIS --- MANISHA

# In[38]:


# store the training dataset in a local variable
dataset_categorical =mydata_cat
dataset_continuous = mydata_con


# In[39]:


plt.figure(1)
plt.subplot(421)
sns.distplot(dataset_categorical['Service time'])

plt.subplot(424)
sns.distplot(dataset_categorical['Work load Average/day '])

plt.subplot(425)
sns.distplot(dataset_categorical['Hit target'])

plt.subplot(428)
sns.distplot(dataset_categorical['Son'])


# In[40]:


# categorical variables
plt.figure(2)
plt.subplot(121)
sns.countplot(dataset_categorical['Disciplinary failure'])

plt.subplot(122)
sns.countplot(dataset_categorical['Education'])


# In[41]:


# plot between dependent and independent variable

subset = dataset_continuous.groupby('Disciplinary failure')['Absenteeism time in hours'].sum()

fig = plt.figure()

ax = fig.add_subplot(111)
subset.plot(kind='bar')

ax.set_ylabel('Sum of Absenteeism time in hours')


# In[42]:


subset = dataset_continuous.groupby('Education')['Absenteeism time in hours'].sum()

fig = plt.figure()

ax = fig.add_subplot(111)
subset.plot(kind='bar')

ax.set_ylabel('Sum of Absenteeism time in hours')


subset.plot(kind='line')


# In[44]:


hit = dataset_continuous.groupby('Hit target')[['Absenteeism time in hours']].mean()
ax = hit.plot(kind='bar', figsize=(7,4), legend=True)
ax.set_xlabel('hit target')
ax.set_ylabel('Absenteeism time in hours')
ax.set_title('Average Absenteeism time in hours by hit target')
plt.show()


# In[46]:


data_ser = dataset_continuous.groupby('Service time')[['Absenteeism time in hours']].mean()
ax = data_ser.plot(kind='bar', figsize=(7,4), legend=True)
ax.set_xlabel('Service time')
ax.set_ylabel('Absenteeism time in hours')
ax.set_title('Average Absenteeism time in hours by Service time')
plt.show()


# In[47]:


#Consider necessary columns only
# Drop duplicates
df1 = mydata_con
df1 = df1[['ID', 'Reason for absence', 'Month of absence', 
           'Social drinker', 'Social smoker', 'Pet', 'Weight',
           'Height', 'Body mass index', 'Absenteeism time in hours',
          'followUp_req']]
df1.drop_duplicates()
df1.head()


# In[48]:


# Aggregate absenteeism hours by grouping by ID, irrpespective or Reason/Month
df2 = df1.groupby(['ID', 'Social drinker', 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index'],
                   as_index = False).agg({'Absenteeism time in hours': "sum"})
df2.head(10)


# In[49]:


grid = sns.PairGrid(data = df2, 
             vars = ['Social drinker', 'Social smoker', 'Pet', 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours'] )
grid = grid.map_lower(sns.regplot, color = 'blue')
grid = grid.map_diag(plt.hist, bins = 10, color = 'red', 
                     edgecolor = 'k')
grid = grid.map_upper(sns.kdeplot, cmap = 'Reds')


# In[50]:


fig, ax = plt.subplots(1,2)
fig.set_size_inches(15, 8)
sns.factorplot('Social drinker',
               'Absenteeism time in hours',
               'Social drinker',
               data=df2,
               kind="box",
               size=6, aspect=0.5, ax=ax[0])

sns.factorplot('Social smoker',
               'Absenteeism time in hours',
               'Social smoker',
               data=df2,
               kind="box",
               size=6, aspect=0.5, ax=ax[1])


plt.close(2)
plt.close(3)


# In[51]:


fig, ax = plt.subplots(1,3)
fig.set_size_inches(20, 8)
sns.regplot('Weight', 'Absenteeism time in hours', color = 'darkorange', data=df2, ax=ax[1] )
sns.regplot('Height', 'Absenteeism time in hours', data=df2, ax=ax[2])
sns.scatterplot('Body mass index', 'Absenteeism time in hours', color='red', data=df2, ax=ax[0])


# In[53]:


plt.figure(figsize=(10,8))
sns.set_style(style = "whitegrid")
count_absents_pet = pd.DataFrame(df2.groupby('Pet',as_index = False).agg({'Absenteeism time in hours':'sum'}))
sns.barplot('Pet','Absenteeism time in hours',data = count_absents_pet, color = 'pink',edgecolor='grey')
plt.title("Absenteeism time for number of pets")


# # DATA MODELLING

# ## DECISION TREE (CATEGORICAL) 

# In[54]:


dataset_categorical =mydata_cat
dataset_continuous = mydata_con

dataset_categorical['Reason for absence'] = dataset_categorical['Reason for absence'].astype('category').cat.codes
dataset_categorical['Month of absence'] = dataset_categorical['Month of absence'].astype('category').cat.codes
dataset_categorical['Day of the week'] = dataset_categorical['Day of the week'].astype('category').cat.codes
dataset_categorical['Seasons'] = dataset_categorical['Seasons'].astype('category').cat.codes
dataset_categorical['Disciplinary failure'] = dataset_categorical['Disciplinary failure'].astype('category').cat.codes
dataset_categorical['Education'] = dataset_categorical['Education'].astype('category').cat.codes
dataset_categorical['Social drinker'] = dataset_categorical['Social drinker'].astype('category').cat.codes
dataset_categorical['Social smoker'] = dataset_categorical['Social smoker'].astype('category').cat.codes
dataset_categorical['Pet'] = dataset_categorical['Pet'].astype('category').cat.codes
dataset_categorical['followUp_req'] = dataset_categorical['followUp_req'].astype('category').cat.codes
dataset_categorical['Absenteeism categories'] = dataset_categorical['Absenteeism categories'].astype('category').cat.codes


# In[55]:


target = dataset_categorical['Absenteeism categories']
categorical = dataset_categorical.drop(['Absenteeism categories'], axis=1)

#categorical = pd.get_dummies(categorical, ['Reason for absence','Month of absence','Day of the week','Seasons','Disciplinary failure','Education','Social drinker','Social smoker','Pet','followUp_req'])

y_check = mydata_test['Absenteeism categories']
categorical_check = mydata_test.drop(['Absenteeism categories','Absenteeism time in hours'], axis=1)


# In[56]:


numbers = LabelEncoder()
y = numbers.fit_transform(target.astype('str'))
y_check = numbers.fit_transform(y_check.astype('str'))
X_train,X_test,y_train,y_test = train_test_split(categorical, y, test_size=.10, random_state=1)

count_original =target.value_counts()
print("count in original dataset ", count_original )

clf = DecisionTreeClassifier(criterion='gini', max_depth=7)

clf = clf.fit(X_train,y_train)

y_predict = clf.predict(X_test)

acc = accuracy_score(y_test,y_predict)*100

print("Accuracy of decision tree with categorical target variable is [validation set]:",acc)
y_predict_test = clf.predict(categorical_check)

acc_test = accuracy_score(y_check,y_predict_test)*100

print('Decision tree accuracy of test set: ',acc_test)


# In[57]:


print("Classification report : ")
print(classification_report(y_test, y_predict))


# In[58]:


# Feature selection through correlation

col = ['ID','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Hit target','Weight','Height','Body mass index']

data_cor = dataset_categorical.loc[:,col]
corr = data_cor.corr()


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin= -1, vmax=1)
fig.colorbar(cax)

# arange is a tool for creating numeric sequences
tick = np.arange(0,len(data_cor.columns),1)

ax.set_xticks(tick)
ax.set_yticks(tick)

plt.xticks(rotation=90)

ax.set_xticklabels(col)
ax.set_yticklabels(col)

plt.show()


# In[59]:


categorical_features = dataset_categorical.drop(['Weight','Age','Absenteeism categories'], axis=1)

#categorical_features = pd.get_dummies(categorical_features, ['Reason for absence','Month of absence','Day of the week','Seasons','Disciplinary failure','Education','Social drinker','Social smoker','Pet','followUp_req'])

X_train,X_test,y_train,y_test = train_test_split(categorical_features, y, test_size=.10, random_state=1)

clf_sel = DecisionTreeClassifier(criterion='gini', max_depth=7)

clf_sel = clf_sel.fit(X_train,y_train)

y_predict_sel = clf_sel.predict(X_test)

acc_sel = accuracy_score(y_test,y_predict_sel)*100

print("Accuracy of decision tree with categorical target variable after features selection is [validation set]:",acc_sel)

categorical_check_sel = categorical_check.drop(['Weight','Age'], axis=1)
y_predict_test_sel = clf_sel.predict(categorical_check_sel)

acc_test_sel = accuracy_score(y_check,y_predict_test_sel)*100

print('Decision tree accuracy of test set after features selection: ',acc_test_sel)


# In[60]:


# Categorical model after scaling

#categorical_scaled = StandardScaler().fit_transform(categorical_features)
categorical_scaled = scale(categorical_features)
X_train,X_test,y_train,y_test = train_test_split(categorical_scaled, y, test_size=.10, random_state=1)

clf_scaled = DecisionTreeClassifier(criterion='gini', max_depth=4)

clf_scaled = clf_scaled.fit(X_train,y_train)

y_predict_scaled = clf_scaled.predict(X_test)

acc_scaled = accuracy_score(y_test,y_predict_scaled)*100

print("Accuracy of decision tree after scaling [validation set]",acc_scaled)

categorical_check_scaled =scale(categorical_check_sel)
y_predict_test_scaled = clf_scaled.predict(categorical_check_scaled)

acc_test_scaled = accuracy_score(y_check,y_predict_test_scaled)*100

print('Decision tree accuracy of test set after features selection: ',acc_test_scaled)


# In[61]:


# Categorical model with pca
from sklearn.decomposition import PCA

pca = PCA(n_components=5)

categorical_pca = pca.fit_transform(categorical_features)
print('variance ratio',pca.explained_variance_ratio_.cumsum())

categorical_pca_df = pd.DataFrame(data=categorical_pca, columns=['pc1','pc2','pc3','pc4','pc5'])

X_train,X_test,y_train,y_test = train_test_split(categorical_pca_df, y, test_size=.10, random_state=1)

clf_pca = DecisionTreeClassifier(criterion='entropy', max_depth=6)

clf_pca = clf_pca.fit(X_train,y_train)

y_predict_pca = clf_pca.predict(X_test)

acc_pca = accuracy_score(y_test,y_predict_pca)*100

print("Accuracy of decision tree with PCA [validation set]: ",acc_pca)

categorical_pca_test = pca.fit_transform(categorical_check_scaled)
print('variance ratio',pca.explained_variance_ratio_.cumsum())

categorical_pca_df_test = pd.DataFrame(data=categorical_pca_test, columns=['pc11','pc12','pc13','pc14','pc15'])
y_predict_test_pca = clf_pca.predict(categorical_pca_df_test)

acc_test_pca = accuracy_score(y_check,y_predict_test_pca)*100

print('Decision tree accuracy of test set after pca: ',acc_test_pca)


# In[62]:


print(" So best accuracy achieved by using decision tree for categorical target variable is after doing feature selection and scaling. Following are confusion matrix and classification report for the same.")


# In[63]:


print("Classification report : ")
print(classification_report(y_check,y_predict_test_sel))


# In[64]:


confusion_matrix(y_check,y_predict_test_sel)


# In[170]:


print("********************* DECISION TREE ***********************")

print("\n################# Training Data Accuracies #####################\n")
print("DECISION TREE - 49.25% ")

print("\n################### Test Data Accuracies ########################\n")

print("DECISION TREE - 39.18%")
print("\n#################################################################\n")


# ## K neighbors Categorical target variable

# In[65]:


X_train,X_test,y_train,y_test = train_test_split(categorical_scaled, y, test_size=.10, random_state=1)

from sklearn.neighbors import KNeighborsClassifier 
neigh = KNeighborsClassifier(n_neighbors=23).fit(X_train, y_train) 

y_predict_k = neigh.predict(X_test)

acc_knn = accuracy_score(y_test,y_predict_k)*100

print("Accuracy of KNN after scaling [validation set]",acc_knn)

y_predict_test_k = neigh.predict(categorical_check_scaled)

acc_test_knn = accuracy_score(y_check,y_predict_test_k)*100

print('KNN accuracy of test set: ',acc_test_knn)


# In[171]:


print("Classification report : ")
print(classification_report(y_check,y_predict_test_k))


# In[172]:


print("********************* knn  ***********************")

print("\n################# Training Data Accuracies #####################\n")
print("KNN - 44.77% ")
print("\n################### Test Data Accuracies ########################\n")
print("KNN - 40.54%")
print("\n#################################################################\n")


# ## RANDOM FOREST 

# In[38]:


#read the dataset
#mydata_cat = pd.read_csv('cleanDataset_categoricalTarget.csv')


# ### USER DEFINED FUNCTIONS

# In[113]:


random_state_split = 121 
#function to split data
def split_data(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.05,random_state = random_state_split)
    return(x_train,x_test,y_train,y_test)


# In[114]:


# function for modelling
def model_fit(x,y,randomforest):
    model_rf = randomforest.fit(x,y)
    return model_rf


# In[115]:


# function for model evaluation
def model_eval(fit,x,y):
    # predict on test data
    y_pre = fit.predict(x)

    # evaluate the model
    acc = accuracy_score(y_pre,y)
    print("Accuracy of validation test set:{}%".format(round(acc*100,2)))

    #print("Confusion Matrix\n",confusion_matrix(y_pre,y))


# In[116]:


#  define the response and target varaibles
features = mydata_cat.drop(['Absenteeism categories'],axis = 1)
target = mydata_cat['Absenteeism categories']
target.value_counts()


# In[117]:


# split the data
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size = 0.2,random_state = 121)

# train model
randomforest = RandomForestClassifier(random_state = 100)
model = model_fit(x_train, y_train,randomforest) 

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# ### FEATURE SCALING

# In[118]:


# standardize the varaibles
features_scaled = StandardScaler().fit_transform(features)
#split the data
x_train,x_test,y_train,y_test = train_test_split(features_scaled,target,test_size = 0.2,random_state = 121)

# train model
randomforest = RandomForestClassifier(random_state = 100)
model = model_fit(x_train, y_train,randomforest) 

print("RESULTS OF SCALING THE DATA: ")

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# ### DATA SIZE INCREASE

# In[119]:


# standardize the varaibles
features_scaled = StandardScaler().fit_transform(features)
#split the data
x_train,x_test,y_train,y_test = split_data(features_scaled,target)

# train model
randomforest = RandomForestClassifier(random_state = 100)
model = model_fit(x_train, y_train,randomforest) 

print("RESULTS AFTER INCREASING THE DATASET SIZE: ")

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# ### PARAMETR TUNNING

# In[75]:


# train model
randomforest = RandomForestClassifier(n_estimators=1000, min_samples_leaf=2, n_jobs= -1,oob_score = -1,warm_start = True,
                                     criterion = 'gini',random_state = 100)
model = model_fit(x_train, y_train,randomforest) 

print("RESULTS WITH PARAMATER TUNNING: ")

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# In[120]:


# train model
randomforest = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1,warm_start = True,
                                    criterion = 'gini',min_samples_split=2,random_state = 100)
model = model_fit(x_train, y_train,randomforest) 

print("RESULTS WITH PARAMATER TUNNING: ")

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# ### FEATURE SELECTION

# In[121]:


# Get numerical feature importances
importances = list(randomforest.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(feature_importances)


# In[122]:


# List of features for later use
feature_list = list(features.columns)

# list of x locations for plotting
x_values = list(range(len(importances)))
plt.figure(figsize = (10,5))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'b', edgecolor = 'k', linewidth = 1.2)

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation=90)

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[123]:


# Fit model using each importance as a threshold
thresholds = sorted(randomforest.feature_importances_)
print(thresholds)


for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(randomforest, threshold=thresh, prefit=True)
    
    select_X_train = selection.transform(x_train)
    #select_X_train = selection.transform(features_scaled)

    # train model
    selection_model = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1,oob_score = True,warm_start = True,
                                     criterion = 'gini',min_samples_split=2,random_state = 100)
    selection_model.fit(select_X_train, y_train)
    
    # eval model
    select_X_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_X_test)
    
    predictions = [value for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    #accuracy = accuracy_score(target_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# #### TEST VALIDATION FOR RANDOM FOREST

# In[124]:


#read the test dataset
mydata_cat_test = pd.read_csv("cleanDataset_categoricalTarget_test.csv")


# In[125]:


#test varaibles
#mydata_cat_test = mydata_test
features_test = mydata_cat_test.drop(['Absenteeism categories'],axis = 1)
target_test = mydata_cat_test['Absenteeism categories']


# In[126]:


#scaling the data
features_scaled_test = StandardScaler().fit_transform(features_test)
randomforest = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1,oob_score = True,warm_start = True,
                                    criterion = 'gini',min_samples_split=2,random_state = 100)

model = randomforest.fit(features_scaled,target)

y_predict = model.predict(features_scaled_test)
print("RANDOM FOREST TEST ACCURACY:{}%".format(round(accuracy_score(y_predict,target_test)*100,2)))


# In[127]:


selection = SelectFromModel(randomforest,threshold = 0.011,prefit = True)# 0.035
select_x_train = selection.transform(features_scaled)

selection_model = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1,oob_score = True,warm_start = True,
                                     criterion = 'gini',min_samples_split=2,random_state = 123)
sel_model = selection_model.fit(select_x_train,target)

select_x_test = selection.transform(features_scaled_test)
y_pre = sel_model.predict(select_x_test)
print("RANDOM FOREST TEST ACCURACY WITH FEATURE SELECTION:{}%".format(round(accuracy_score(y_pre,target_test)*100,2)))


# ### PCA with RANDOM FOREST

# In[128]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 8)
pc = pca.fit(features_scaled)
pc.explained_variance_ratio_.cumsum()


# In[129]:


# pca features
new_features = pca.fit_transform(features_scaled)
# split the dataset
x_train,x_test,y_train,y_test = split_data(new_features,target)

randomforest = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1,oob_score = True,warm_start = True,
                                     criterion = 'gini',min_samples_split=2,random_state = 100)

model = model_fit(x_train, y_train,randomforest) 

print("RESULTS WITH PCA: ")

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# In[130]:


# PCA on test set
new_features = pca.fit_transform(features_scaled_test)

y_predict = model.predict(new_features)
print("Accuracy of test set:{}%".format(round(accuracy_score(y_predict,target_test)*100,2)))


# ### LDA with RANDOM FOREST

# In[131]:


# LDA features
lda = LinearDiscriminantAnalysis(n_components=3) 
# scaled 
new_features = lda.fit(features_scaled,target).transform(features_scaled)
# split the data
x_train,x_test,y_train,y_test = split_data(new_features,target)

randomforest = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, n_jobs=-1,oob_score = True,warm_start = True,
                                     criterion = 'gini',min_samples_split=2,random_state = 100)


model = model_fit(x_train, y_train,randomforest) 

print("RESULTS WITH LDA: ")

# check accuracy
print(model)
y_predict = model.predict(x_train)
print("Accuracy of training set:{}%".format(round(accuracy_score(y_predict,y_train)*100,2)))
y_predict = model_eval(model,x_test,y_test)


# In[132]:


# LDA on test set
new_features = lda.fit(features_scaled_test,target_test).transform(features_scaled_test)

y_predict = model.predict(new_features)
print("Accuracy of test set:{}%".format(round(accuracy_score(y_predict,target_test)*100,2)))


# In[133]:


print("********************* RANDOM FOREST ***********************")

print("\n################# Training Data Accuracies #####################\n")
print("RANDOM FOREST - 43.28% ")
print("RANDOM FOREST PARAMETER TUNNING - 61.76")
print("RANDOM FOREST FEATURE SELECTION - 64.71%")

print("RANDOM FOREST with PCA - 50.0%")
print("RANDOM FOREST with LDA - 58.82% ")

print("\n################### Test Data Accuracies ########################\n")

print("RANDOM FOREST PARAMETER TUNNING - 44.59")
print("RANDOM FOREST FEATURE SELECTION - 47.3%")

print("RANDOM FOREST with PCA - 33.78%")
print("RANDOM FOREST with LDA - 44.59%")
print("\n#################################################################\n")


# ## EXTREME GRADIENT BOOSTING

# In[96]:


# import categorical target variable dataset
mydata_cat = pd.read_csv("cleanDataset_categoricalTarget_1.csv")
# test dataset
mydata_cat_test = pd.read_csv("cleanDataset_categoricalTarget_test.csv")


features = mydata_cat.drop(['Absenteeism categories'],axis = 1)
target = mydata_cat['Absenteeism categories']
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(target)

#test varaibales
features_test = mydata_cat_test.drop(['Absenteeism categories'],axis = 1)
target_test = mydata_cat_test['Absenteeism categories']
# encode string class values as integers
label_encoded_y_test = LabelEncoder().fit_transform(target_test)


# In[97]:


#split dataset
x_train,x_test,y_train,y_test = train_test_split(features,label_encoded_y,test_size = 0.05,random_state = 121)

# fit model no training data
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print(xgb)

# make predictions for test data
y_pred = xgb.predict(x_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of XGBM on validataion test: %.2f%%" % (accuracy * 100.0))


# In[99]:


# model for test set ,on all train set
model_test = xgb.fit(features,label_encoded_y)
#print(model)

y_predict = model_test.predict(features_test)
print("Accuracy of XGBOOST test set:{}%".format(round(accuracy_score(y_predict,label_encoded_y_test)*100,2)))


# #### Feature Selection

# In[100]:


thresholds = sorted(xgb.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(xgb, threshold=thresh, prefit=True)
    select_X_train = selection.transform(x_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# ### Best Thresh=0.046, n=11, Accuracy: 70.59%

# In[101]:


# Test set model for feature selection
selection = SelectFromModel(xgb,threshold = 0.047,prefit = True)
select_x_train = selection.transform(features)

selection_model =  XGBClassifier()
sel_model = selection_model.fit(select_x_train,label_encoded_y)

select_x_test = selection.transform(features_test)
y_pre = sel_model.predict(select_x_test)
print("Accuracy of test set:{}%".format(round(accuracy_score(y_pre,label_encoded_y_test)*100,2)))


# #### Parameter Tuning

# In[102]:


# grid search
model = XGBClassifier()
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

param_grid = dict(max_depth=max_depth, n_estimators=n_estimators,learning_rate = learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(features, label_encoded_y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


# In[103]:


# grid search with 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 50
model = XGBClassifier()
n_estimators = [50]
max_depth = [4]
learning_rate = [0.1]

# kfold
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators,learning_rate = learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(features, label_encoded_y)

# predict 
y_pre = grid_result.predict(features)
print("Accuracy of validation set:{}%".format(round(accuracy_score(y_pre,label_encoded_y)*100,2)))


# #### TEST SET VALIDATION 

# In[104]:


# Test set model for feature selection
selection = SelectFromModel(xgb,threshold = 0.047,prefit = True)
select_x_train = selection.transform(features)

selection_model =  XGBClassifier()
sel_model = selection_model.fit(select_x_train,label_encoded_y)


y_pre = grid_result.predict(features_test)
print("Accuracy of test set:{}%".format(round(accuracy_score(y_pre,label_encoded_y_test)*100,2)))


# In[105]:


print("********************* XGBOOST ***********************")
print("\n################# Training Data Accuracies #######################\n")
print("XGBOOST - 61.76% ")
print("XGBOOST FEATURE SELECTION - 70.59")
print("XGBOOST PARAMETER TUNNING - 74.02%")

print("\n################### Test Data Accuracies ########################\n")

print("XGBOOST - 45.95% ")
print("XGBOOST FEATURE SELECTION - 56.76%")
print("XGBOOST PARAMETER TUNNING - 48.65%")
print("\n#################################################################\n")


# ## LOGISTIC REGRESSION 

# In[143]:


# Train multi-classification model with logistic regression
mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',warm_start = True,n_jobs = -1).fit(x_train,y_train)

print ("Multinomial Logistic regression validation Accuracy = ", round(accuracy_score(y_test, mul_lr.predict(x_test))*100,2), "%")


# In[144]:


# test result
label_encoded_y = LabelEncoder().fit_transform(target)
label_encoded_y_test = LabelEncoder().fit_transform(target_test)
mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',warm_start = True,n_jobs = -1).fit(features,label_encoded_y)
acc = accuracy_score(label_encoded_y_test, mul_lr.predict(features_test))
print ("Multinomial Logistic regression test Accuracy = ", round(acc*100,2), "%")


# In[145]:


print("********************* LOGISTIC REGRESSION ***********************")
print("\n################# Training Data Accuracies #######################\n")
print("LGC - 58.82% ")

print("\n################### Test Data Accuracies ########################\n")
print("LGC - 45.95% ")
print("\n#################################################################\n")


# ## SUPPORT VECTOR MACHINE 

# In[146]:


# laod the data
X = mydata_cat.drop(['Absenteeism categories'], axis = 1)
y = mydata_cat['Absenteeism categories'].astype('category')

# Scale the data and apply pca 
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_.sum()


# In[ ]:


#Store Principal components in a different frame
p_df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pc4',
                                                                  'pc5','pc6','pc7','pc8','pc9','pc10'])


# SVM with LINEAR Kernel
X_train, X_test, y_train, y_test = train_test_split(p_df, y, test_size=0.25, random_state=1234)
clf = svm.SVC(kernel='linear') 
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
accuracy_score(y_test, y_predict)*100


# In[149]:


# Cross Validation - SVM - linear
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
model = svm.SVC(kernel='linear')

#PCA Data frame 
results1 = cross_val_score(model, p_df, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))

from sklearn.preprocessing import scale
dat = X
data_normal = scale(dat)
 # Data Scaled using scale() instead of standard scaler
results2 = cross_val_score(model, data_normal, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean()*100.0, results2.std()*100.0))

 # without scaling / pca - very slow
# results3 = model_selection.cross_val_score(model, X, y, cv=kfold)
# print("Accuracy: %.3f%% (%.3f%%)" % (results3.mean()*100.0, results3.std()*100.0))


# In[150]:


# Cross Validation - SVM - Polynomial Kernel
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
# n_splits 3-44, 7-43, 10-46, but std dev was high, so retaining n splits = 5
model = svm.SVC(kernel='poly', degree = 1)

#PCA Data frame 
results1 = cross_val_score(model, p_df, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))


# In[151]:


# Cross Validation - SVM - poly - on unscaled data with no pca - no improvement
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
# n_splits 3-44, 7-43, 10-46, but std dev was high, so retaining n splits = 5
model = svm.SVC(kernel='poly', degree = 1)

#PCA Data frame 
results1 =cross_val_score(model, X, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))


# In[152]:


# Cross Validation - SVM - poly - on unscaled data with no pca - no improvement
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
# n_splits 3-44, 7-43, 10-46, but std dev was high, so retaining n splits = 5
model = svm.SVC(kernel='poly', degree = 1)

#PCA Data frame 
results1 = cross_val_score(model, X, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))


# In[1]:


print("********************* SUPPORT VECTOR MACHINE ***********************")

print("\n################# Training Data Accuracies #####################\n")
print("SVM-LINEAR - 49.25% ")
print("SVM-POLYNOMIAL - 49.25% ")

print("\n################### Test Data Accuracies ########################\n")
print("SVM-LINEAR - 49.25% ")
print("SVM-POLYNOMIAL - 49.25% ")
print("\n#################################################################\n")


# ### NAIVE BAYES 

# In[210]:


# read the data
mytest_cat = pd.read_csv("cleanDataset_categoricalTarget_test.csv")
train_data = mydata_cat.copy()
test_data = mytest_cat.copy()

test_size=0.10
random_state = 123
cv = 8
gamma = 'auto'


# In[203]:


## Data Preprocessing for NB

# Rename Columns
test_data.columns = test_data.columns.str.replace("Work load Average/day","Workload")
train_data.columns = train_data.columns.str.replace("Work load Average/day","Workload")

train_data = train_data.rename(index=str, columns={
    "Reason for absence": "Reason", "Month of absence": "Month", "Day of the week": "Day", 
    "Transportation expense": "Transport","Distance from Residence to Work" : "Distance", 
    "Absenteeism categories": "Categories", "followUp_req": "FollowUp","Hit target": "Target", 
    "Disciplinary failure": "Disciplinary", "Service time" : "Service", "Social drinker": "Drinker",
    "Social smoker": "Smoker", "Workload ": "Workload", "Body mass index": "BMI"})

test_data = test_data.rename(index=str, columns={
    "Reason for absence": "Reason", "Month of absence": "Month", "Day of the week": "Day", 
    "Transportation expense": "Transport","Distance from Residence to Work" : "Distance", 
    "Absenteeism categories": "Categories", "followUp_req": "FollowUp","Hit target": "Target", 
    "Disciplinary failure": "Disciplinary", "Service time" : "Service", "Social drinker": "Drinker",
    "Social smoker": "Smoker", "Workload ": "Workload", "Body mass index": "BMI"})


# In[204]:


#Dropping Month, Day, Seasons since absenteeism since these are not good predictors for absenteeism and 
# from EDA we did not find much correlation between them and absenteeism
# Dropping Height and Weight since we have BMI column
# Dropping FollowUp column which we added
train_data = train_data.drop(['Month', 'Day', 'Weight', 'Height', 'Seasons', 'FollowUp'], axis = 1)
test_data = test_data.drop([ 'Month', 'Day', 'Weight','Height', 'Seasons', 'FollowUp'], axis = 1)

# Cleaning data by removing outliers
train_data = train_data[train_data.Target != 0]
test_data = test_data[test_data.Target != 0]


# In[205]:


# Making a copy of preprocessed data for SVM model analysis
# For NB analysis, we convert all columns to categorical data (ordinal and nominal)
train_data_svm = train_data.copy()
test_data_svm = test_data.copy()


# In[206]:


# Convert to ordered categorical data 
target_iqr = mquantiles(train_data['Target'])
distance_iqr = mquantiles(train_data['Distance'])
service_iqr = mquantiles(train_data['Service'])
age_iqr = mquantiles(train_data['Age'])
workload_iqr = mquantiles(train_data['Workload'])
bmi_iqr = mquantiles(train_data['BMI'])
transport_iqr = mquantiles(train_data['Transport'])

def classify_iqr(colname, df, iqr):
    new_col = []
    for value in df[colname]:
        if value < iqr[0]:
            new_col.append(0)
        elif iqr[0] <= value < iqr[1]:
            new_col.append(1)
        elif iqr[1] <= value < iqr[2]:
            new_col.append(2)
        else:
            new_col.append(3)
    return new_col
            
test_data.Workload = classify_iqr("Workload", test_data, workload_iqr)
test_data.Target = classify_iqr("Target", test_data, target_iqr)
test_data.Service = classify_iqr("Service", test_data, service_iqr)
test_data.Age = classify_iqr("Age", test_data, age_iqr)
test_data.BMI = classify_iqr("BMI", test_data, bmi_iqr)
test_data.Distance = classify_iqr("Distance", test_data, distance_iqr)
test_data.Transport = classify_iqr("Transport", test_data, transport_iqr)


train_data.Workload = classify_iqr("Workload", train_data, workload_iqr)
train_data.Target = classify_iqr("Target", train_data, target_iqr)
train_data.Service = classify_iqr("Service", train_data, service_iqr)
train_data.Age = classify_iqr("Age", train_data, age_iqr)
train_data.BMI = classify_iqr("BMI", train_data, bmi_iqr)
train_data.Distance = classify_iqr("Distance", train_data, distance_iqr)
train_data.Transport = classify_iqr("Transport", train_data, transport_iqr)


# In[207]:


# Reason and ID are not ordered, rest of the columns are ordered.

train_data["ID"] = train_data["ID"].astype("category")
test_data["ID"] = test_data["ID"].astype("category")

train_data["Reason"] = train_data["Reason"].astype("category")
test_data["Reason"] = test_data["Reason"].astype("category")

train_data["Categories"] = train_data["Categories"].astype("category")
test_data["Categories"] = test_data["Categories"].astype("category")
train_data['Categories'].cat.reorder_categories(['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6'])
test_data['Categories'].cat.reorder_categories(['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6'])
train_data['Categories'] = train_data.Categories.astype("category").cat.codes
test_data['Categories'] = test_data.Categories.astype("category").cat.codes


# In[208]:


X = train_data.drop(['Categories'], axis = 1)
y = train_data["Categories"]

X_test_data = test_data.drop(['Categories'], axis = 1)
y_test_data = test_data["Categories"]


X.head()


# In[211]:


# naive bayes without feature selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

#############TRAIN DATA###############
gnb = GaussianNB()
model_nb = gnb.fit(X_train, y_train)
y_predict = model_nb.predict(X_test)

cm_nb = confusion_matrix(y_test, y_predict)
accuracy_nb = accuracy_score(y_test, y_predict)*100

scores_nb = cross_val_score(model_nb, X, y, cv=cv)
predictions_nb = cross_val_predict(model_nb, X, y, cv=cv)
accuracy_cv_nb = metrics.accuracy_score(y, predictions_nb)

#############TEST DATA###############
gnb = GaussianNB()
model = gnb.fit(X, y)
y_predict_test = model.predict(X_test_data)
cm_test_nb = confusion_matrix(y_test_data, y_predict_test)
accuracy_test_nb = accuracy_score(y_test_data, y_predict_test)*100                   


# In[212]:


df = X
df_test = X_test_data
corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()


# In[213]:


# Create correlation matrix 
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.3)]
Xnew = df.drop(to_drop, axis=1)

print(Xnew.info())


# In[214]:


Xnew_test_data = df_test.drop(to_drop, axis=1)
# naive bayes with feature selection
Xnew_train, Xnew_test, ynew_train, ynew_test = train_test_split(Xnew, y, test_size=test_size, random_state=random_state)

#############TRAIN DATA###############
gnb = GaussianNB()
model_nb_fs = gnb.fit(Xnew_train, ynew_train)
ynew_predict = model_nb_fs.predict(Xnew_test)

cm_nb_fs = confusion_matrix(ynew_test, ynew_predict)
accuracy_nb_fs = accuracy_score(ynew_test, ynew_predict)*100

scores_nb_fs = cross_val_score(model_nb_fs, Xnew, y, cv=cv)
predictions_nb_fs = cross_val_predict(model_nb_fs, Xnew, y, cv=cv)
accuracy_cv_nb_fs = metrics.accuracy_score(y, predictions_nb_fs)

#############TEST DATA###############
gnb = GaussianNB()
model = gnb.fit(Xnew, y)
ynew_predict_test = model.predict(Xnew_test_data)
cm_test_nb_fs = confusion_matrix(y_test_data, ynew_predict_test)
accuracy_test_nb_fs = accuracy_score(y_test_data, ynew_predict_test)*100


# ## SVM with RBF Kernel

# In[216]:


# svm with rbf with feature selection
Xnew_train, Xnew_test, ynew_train, ynew_test = train_test_split(Xnew, y, test_size=test_size, random_state=random_state)

#############TRAIN DATA###############
clf = SVC(kernel='rbf', gamma = gamma)
model_rbf_fs = clf.fit(Xnew_train, ynew_train)
ynew_predict = model_rbf_fs.predict(Xnew_test)

cm_rbf_fs = confusion_matrix(ynew_test, ynew_predict)
accuracy_rbf_fs = accuracy_score(ynew_test, ynew_predict)*100

scores_rbf_fs = cross_val_score(model_rbf_fs, Xnew, y, cv=cv)
predictions_rbf_fs = cross_val_predict(model_rbf_fs, Xnew, y, cv=cv)
accuracy_cv_rbf_fs = metrics.accuracy_score(y, predictions_rbf_fs)

#############TEST DATA###############
clf = SVC(kernel='rbf', gamma = gamma)
model = clf.fit(Xnew, y)
ynew_predict_test = model.predict(Xnew_test_data)
cm_test_rbf_fs = confusion_matrix(y_test_data, ynew_predict_test)
accuracy_test_rbf_fs = accuracy_score(y_test_data, ynew_predict_test)*100 


# In[217]:


train_data_svm["Categories"] = train_data_svm["Categories"].astype("category")
test_data_svm["Categories"] = test_data_svm["Categories"].astype("category")
train_data_svm['Categories'].cat.reorder_categories(['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6'])
test_data_svm['Categories'].cat.reorder_categories(['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6'])
train_data_svm['Categories'] = train_data_svm.Categories.astype("category").cat.codes
test_data_svm['Categories'] = test_data_svm.Categories.astype("category").cat.codes


# In[218]:


X_svm = train_data_svm.drop(['Categories'], axis = 1)
y_svm = train_data_svm["Categories"]

X_test_data_svm = test_data_svm.drop(['Categories'], axis = 1)
y_test_data_svm = test_data_svm["Categories"]


# In[220]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_svm.values)
df_svm = pd.DataFrame(X_scaled, columns = X_svm.columns)
df_svm.head()

X_scaled_test = scaler.fit_transform(X_test_data_svm.values)
df_test_svm = pd.DataFrame(X_scaled_test, columns = X_test_data_svm.columns)
df_test_svm.head()


# In[221]:


# svm rbf after scaling
X_train, X_test, y_train, y_test = train_test_split(df_svm, y_svm, test_size=test_size, random_state=random_state)

#############TRAIN DATA###############
clf = SVC(kernel='rbf', gamma = gamma)
model_rbf = clf.fit(X_train, y_train)
y_predict = model_rbf.predict(X_test)

cm_rbf = confusion_matrix(y_test, y_predict)
accuracy_rbf = accuracy_score(y_test, y_predict)*100
scores_rbf = cross_val_score(model_rbf, df_svm, y_svm, cv=cv)
predictions_rbf = cross_val_predict(model_rbf,df_svm, y_svm, cv=cv)
accuracy_cv_rbf = metrics.accuracy_score(y_svm, predictions_rbf)


#############TEST DATA###############
clf = SVC(kernel='rbf', gamma = gamma)
model = clf.fit(df_svm, y_svm)
y_predict_test = model.predict(df_test_svm)

cm_test_rbf = confusion_matrix(y_test_data_svm, y_predict_test)
accuracy_test_rbf = accuracy_score(y_test_data_svm, y_predict_test)*100


# In[222]:


#Fitting the PCA algorithm with our Data
pca = PCA()
X_transform = pca.fit_transform(df_svm)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()


# In[223]:


# svm (rbf) with pca

pca = PCA(n_components=6)
X_transform = pca.fit_transform(df_svm)
print("Cumulative % of variance captured as # components increase = {}".format(np.cumsum(pca.explained_variance_ratio_)*100))

X_train, X_test, y_train, y_test = train_test_split(X_transform, y_svm, test_size=test_size, random_state=random_state)

#############TRAIN DATA###############
clf = SVC(kernel='rbf', gamma = gamma)
model_rbf_pca = clf.fit(X_train, y_train)
y_predict = model_rbf_pca.predict(X_test)

cm_rbf_pca = confusion_matrix(y_test, y_predict)
accuracy_rbf_pca = accuracy_score(y_test, y_predict)*100

scores_rbf_pca = cross_val_score(model_rbf_pca, X_transform, y_svm, cv=cv)
predictions_rbf_pca = cross_val_predict(model_rbf_pca, X_transform, y_svm, cv=cv)
accuracy_cv_rbf_pca = metrics.accuracy_score(y_svm, predictions_rbf_pca)

#############TEST DATA###############
clf = SVC(kernel='rbf', gamma = 0.02)
testdata_transformed = pca.transform(df_test_svm)

model = clf.fit(X_transform, y_svm)
y_predict_test = model.predict(testdata_transformed)

cm_test_rbf_pca = confusion_matrix(y_test_data_svm, y_predict_test)
accuracy_test_rbf_pca = accuracy_score(y_test_data_svm, y_predict_test)*100 


# In[225]:


## naive bayes with pca
pca = PCA(n_components=6)
X_transform = pca.fit_transform(df_svm)
print("Cumulative % of variance captured as # components increase = {}".format(np.cumsum(pca.explained_variance_ratio_)*100))

#############TRAIN DATA###############
X_train, X_test, y_train, y_test = train_test_split(X_transform, y, test_size=test_size, random_state=random_state)

gnb = GaussianNB()
model_nb_pca = gnb.fit(X_train, y_train)
y_predict = model_nb_pca.predict(X_test)

cm_nb_pca = confusion_matrix(y_test, y_predict)
accuracy_nb_pca = accuracy_score(y_test, y_predict)*100

scores_nb_pca = cross_val_score(model_nb_pca, X_transform, y_svm, cv=cv)
predictions_nb_pca = cross_val_predict(model_nb_pca, X_transform, y_svm, cv=cv)
accuracy_cv_nb_pca = metrics.accuracy_score(y_svm, predictions_nb_pca)


#############TEST DATA###############
gnb = GaussianNB()
testdata_transformed = pca.transform(df_test_svm)
model = gnb.fit(X_transform, y_svm)
y_predict_test = model.predict(testdata_transformed)

cm_test_nb_pca = confusion_matrix(y_test_data, y_predict_test)
accuracy_test_nb_pca = accuracy_score(y_test_data, y_predict_test)*100


# In[226]:


training_accuracy = {}
training_accuracy["nb"] = accuracy_nb.round(2)
training_accuracy["nb_fs"] = accuracy_nb_fs.round(2)
training_accuracy["nb_pca"] = accuracy_nb_pca.round(2)

training_accuracy["rbf"] = accuracy_rbf.round(2)
training_accuracy["rbf_fs"] = accuracy_rbf_fs.round(2)
training_accuracy["rbf_pca"] = accuracy_rbf_pca.round(2)


test_accuracy = {}
test_accuracy["nb"] = accuracy_test_nb.round(2)
test_accuracy["nb_fs"] = accuracy_test_nb_fs.round(2)
test_accuracy["nb_pca"] = accuracy_test_nb_pca.round(2)

test_accuracy["rbf"] = accuracy_test_rbf.round(2)
test_accuracy["rbf_fs"] = accuracy_test_rbf_fs.round(2)
test_accuracy["rbf_pca"] = accuracy_test_rbf_pca.round(2)

cv_scores = {}
cv_scores["nb"] = accuracy_cv_nb.round(2)
cv_scores["nb_fs"] = accuracy_cv_nb_fs.round(2)
cv_scores["nb_pca"] = accuracy_cv_nb_pca.round(2)

cv_scores["rbf"] = accuracy_cv_rbf.round(2)
cv_scores["rbf_fs"] = accuracy_cv_rbf_fs.round(2)
cv_scores["rbf_pca"] = accuracy_cv_rbf_pca.round(2)



print("\n#################Training Data Accuracies are######################\n")
print("NB - {}%".format(training_accuracy['nb']))
print("NB with Correlation - {}%".format(training_accuracy['nb_fs']))
print("NB with PCA - {}%".format(training_accuracy['nb_pca']))

print("SVM(RBF) - {}%".format(training_accuracy['rbf']))
print("SVM(RBF) with Correlation - {}%".format(training_accuracy['rbf_fs']))
print("SVM(RBF) with PCA - {}%".format(training_accuracy['rbf_pca']))

print("\n#################Test Data Accuracies are######################\n")

print("NB - {}%".format(test_accuracy['nb']))
print("NB with Correlation - {}%".format(test_accuracy['nb_fs']))
print("NB with PCA - {}%".format(test_accuracy['nb_pca']))

print("SVM(RBF) - {}%".format(test_accuracy['rbf']))
print("SVM(RBF) with Correlation - {}%".format(test_accuracy['rbf_fs']))
print("SVM(RBF) with PCA - {}%".format(test_accuracy['rbf_pca']))


# ## CROSSFOLD VALIDATION

# In[182]:


x_train,x_test,y_train,y_test = train_test_split(features_scaled,target,test_size = 0.05,random_state = 121)


# In[187]:


kfold = StratifiedKFold(n_splits=10)

random_state = 123
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())


cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")




# In[188]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(x_train,y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_


# ## CONTINOUS TARGET

# ### DECISION TREE (CONTINOUS) 

# In[153]:


y_target = dataset_continuous['Absenteeism time in hours']
dataset_cont = dataset_continuous.drop(['Absenteeism time in hours'], axis=1)
#data_cont = pd.get_dummies(dataset_cont, ['Reason for absence','Month of absence','Day of the week','Seasons','Disciplinary failure','Education','Social drinker','Social smoker','Pet','followUp_req'])
dataset_cont = scale(dataset_cont)

X_train,X_test,y_train,y_test = train_test_split(dataset_cont, y_target, test_size=.10, random_state=151)


# In[154]:


# Continuous data model

clf_cont = DecisionTreeRegressor(random_state = 1)
clf_cont = clf_cont.fit(X_train,y_train)

y_predict = clf_cont.predict(X_test)

err= mean_absolute_error(y_test,y_predict)

print('Mean Absolute error for decision tree regressor[validation set]:',err)


rmse = mean_squared_error( y_predict, y_test)**0.5

print('Root mean squared error for decision tree regressor[validation set]',rmse)

y_chck = mydata_test['Absenteeism time in hours']

y_predict_chck = clf_cont.predict(categorical_check)

err_test= mean_absolute_error(y_chck,y_predict_chck)

print('Mean Absolute error for decision tree regressor[test set]:',err_test)


rmse_test = mean_squared_error( y_predict_chck, y_chck)**0.5

print('Root mean squared error for decision tree regressor[test set]',rmse_test)


# In[155]:


from matplotlib import pyplot as plt
plt.figure(figsize=(6, 3))
plt.scatter(y_chck, y_predict_chck)
plt.plot([0, 100], [0, 100], '--k')


# In[156]:


# Decision tree regression using cross validation


scoring = make_scorer(mean_squared_error)
reg_cv = GridSearchCV(DecisionTreeRegressor(random_state=0),
              param_grid={'min_samples_split': range(2, 10),'max_features': ['sqrt', 'log2', None]},
              scoring=scoring, cv=10, refit=True)

reg_cv.fit(X_train, y_train)

print(reg_cv.best_params_)
mean_err =mean_absolute_error(y_test, reg_cv.best_estimator_.predict(X_test))

print('Mean Absolute error of decision tree regressor [validation set]',mean_err)

rmse = mean_squared_error(y_test, reg_cv.best_estimator_.predict(X_test))**0.5

print('root mean square error of decision tree regressor [validation set]',rmse)


err_test_cv= mean_absolute_error(y_chck,reg_cv.best_estimator_.predict(categorical_check))

print('Mean Absolute error for decision tree regressor[test set]:',err_test_cv)


rmse_test_cv = mean_squared_error( y_predict_chck, reg_cv.best_estimator_.predict(categorical_check))**0.5

print('Root mean squared error for decision tree regressor[test set]',rmse_test_cv)


# ### SVM (Continous)

# In[157]:


#Seperate explanatory and target variable
X = mydata_con.drop(['Absenteeism time in hours'], axis = 1)
y = mydata_con['Absenteeism time in hours']


# In[158]:


# Scale the data and apply pca 
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_.sum()


# In[159]:


#make a pc dataframe
p_df = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2', 'pc3', 'pc4',
                                                                  'pc5','pc6','pc7','pc8','pc9','pc10'])
#SVM - linear model
X_train, X_test, y_train, y_test = train_test_split(p_df, y, test_size=0.25, random_state=1234)
clf = svm.SVC(kernel='linear') 
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
accuracy_score(y_test, y_predict)*100


# In[160]:


#Factor Analysis - 

dat = X
data_normal = scale(dat)
fa = FactorAnalysis(n_components = 10)
fa = fa.fit(data_normal)
feature_names = dat.columns
factor_names = ['Factor'+str(x) for x in range(1,11)]
pd.DataFrame(np.transpose(fa.components_), index=feature_names ,
columns=factor_names)


# In[161]:


# Cross Validation - log reg
seed = 7
kfold = KFold(n_splits=5, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, p_df, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[162]:


# Cross Validation - linear reg
#On p_df - Accuracy: 14.096% (4.772%)
#On unscaled data Accuracy: 12.186% (8.636%)
seed = 7
kfold = KFold(n_splits=5, random_state=seed)
model = LinearRegression()
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[163]:


# Cross Validation - SVM - linear
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
model = svm.SVC(kernel='linear')


# In[164]:


#PCA Data frame - 44.7%
results1 = cross_val_score(model, p_df, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))


# In[165]:


# Data Scaled using scale() instead of standard scaler - without PCA
results2 = cross_val_score(model, data_normal, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean()*100.0, results2.std()*100.0))


# In[166]:


# Cross Validation - SVM - rbf
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
model = svm.SVC(kernel='rbf', gamma = 0.1)

#PCA Data frame 
results1 = cross_val_score(model, p_df, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))

# Data Scaled using scale() instead of standard scaler - without PCA
results2 = cross_val_score(model, data_normal, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean()*100.0, results2.std()*100.0))


# In[167]:


# Cross Validation - SVM - poly
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
model = svm.SVC(kernel='poly', degree = 1)

#PCA Data frame 
results1 = cross_val_score(model, p_df, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))

# Data Scaled using scale() instead of standard scaler - without PCA
results2 = cross_val_score(model, data_normal, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results2.mean()*100.0, results2.std()*100.0))


# In[168]:


# Cross Validation - SVM - poly
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
# n_splits 3-44, 7-43, 10-46, but std dev was high, so retaining n splits = 5
model = svm.SVC(kernel='poly', degree = 1)

#PCA Data frame 
results1 = cross_val_score(model, p_df, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))


# In[169]:


# Cross Validation - SVM - poly - on unscaled data with no pca - no improvement
seed = 1234
kfold = KFold(n_splits=5, random_state=seed)
# n_splits 3-44, 7-43, 10-46, but std dev was high, so retaining n splits = 5
model = svm.SVC(kernel='poly', degree = 1)

#PCA Data frame 
results1 = cross_val_score(model, X, y, cv=kfold)  
print("Accuracy: %.3f%% (%.3f%%)" % (results1.mean()*100.0, results1.std()*100.0))


# ## RESULTS 

# #### Top 3 best models for given dataset

# In[94]:


print("################### CATEGORICAL TARGET ##################\n")
results = pd.DataFrame({'Models':["XGBOOST - PARAMETER TUNNING ","XGBOOST-FEATURE SELECTION","SVM(RBF)-PCA"],'ValidationSet':["74.02%","70.59%","56.72%"],'TestSet':["56.76%","48.65%","50.0%"]})
print(results)
print("\n################## CONTINOUS TARGET ###################\n")
result_cont = pd.DataFrame({'Model':['Decision Tree - RMSE'],'ValidationSet':['15.3'],'TestSet':['9.4']})
print(result_cont)
print("\n########################################################\n")

