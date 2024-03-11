#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


df=pd.read_csv('xAPI-Edu-Data.csv')


# # Data Analysis

# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.dropna() 


# In[7]:


# Check if missing values exist or not
df.isnull().sum()


# # EDA

# In[8]:


#plt.subplot(1,2,1)
sns.countplot(x="gender", order=['F','M'], data=df, palette="Set1")
plt.show()


# In[9]:


#plt.subplot(1,2,2)
sns.countplot(x="gender", order=['F','M'], hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[10]:


#plt.subplot(1,2,1)
df['NationalITy'].value_counts(normalize=True).plot(kind='bar')
plt.show()


# In[11]:


#plt.subplot(1,2,2)
df['PlaceofBirth'].value_counts(normalize=True).plot(kind='bar')
plt.show()


# In[12]:


#plt.subplot(1,2,1)
sns.countplot(y="NationalITy", data=df, palette="muted")
plt.show()


# In[13]:


#plt.subplot(1,2,2)
sns.countplot(y="NationalITy", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[14]:


sns.countplot(x="Relation", order=['Mum','Father'], data=df, palette="Set1")
plt.show()


# In[15]:


sns.countplot(x="Relation", order=['Mum','Father'], hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[16]:


sns.countplot(x="StageID", data=df, palette="muted")
plt.show()


# In[17]:


sns.countplot(x="StageID", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[18]:


sns.countplot(x="GradeID", data=df, palette="muted")
plt.show()


# In[19]:


sns.countplot(x="GradeID", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[20]:


plt.subplot(1,2,1)
sns.countplot(x="SectionID", order=['A','B','C'], data=df, palette="muted")

plt.subplot(1,2,2)
sns.countplot(x="SectionID", order=['A','B','C'], hue="Class", hue_order=['L','M','H'], data=df, palette="muted")

plt.show()


# In[21]:


plt.subplot(1,2,1)
sns.countplot(y="Topic", data=df, palette="muted")

plt.subplot(1,2,2)
sns.countplot(y="Topic", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")

plt.show()


# In[22]:


#plt.subplot(1,2,1)
sns.countplot(x="ParentschoolSatisfaction", data=df, palette="muted")
plt.show()


# In[23]:


#plt.subplot(1,2,2)
sns.countplot(x="ParentschoolSatisfaction", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[24]:


# Checking dataset 


# In[25]:


plt.figure(figsize=(8, 8))
sns.countplot('Class', data=df)
plt.title('Balanced Classes')
plt.show()


# # Pre-processing

# gender                      480 non-null object
# NationalITy                 480 non-null object
# PlaceofBirth                480 non-null object
# StageID                     480 non-null object
# GradeID                     480 non-null object
# SectionID                   480 non-null object
# Topic                       480 non-null object
# Semester                    480 non-null object
# Relation                    480 non-null object
# 
# ParentAnsweringSurvey       480 non-null object
# ParentschoolSatisfaction    480 non-null object
# StudentAbsenceDays          480 non-null object
# Class                       480 non-null object

# In[26]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()


# In[27]:


df['LGender'] = le.fit_transform(df['gender'])#.values.reshape(-1,1).ravel())
df['LNationalITy'] = le.fit_transform(df['NationalITy'])
df['LPlaceofBirth'] = le.fit_transform(df['PlaceofBirth'])
df['LStageID'] = le.fit_transform(df['StageID'])
df['LGradeID'] = le.fit_transform(df['GradeID'])
df['LSectionID'] = le.fit_transform(df['SectionID'])
df['LTopic'] = le.fit_transform(df['Topic'])
df['LSemester'] = le.fit_transform(df['Semester'])
df['LRelation'] = le.fit_transform(df['Relation'])
df['LParentschoolSatisfaction'] = le.fit_transform(df['ParentschoolSatisfaction'])
df['LParentAnsweringSurvey'] = le.fit_transform(df['ParentAnsweringSurvey'])
df['LStudentAbsenceDays'] = le.fit_transform(df['StudentAbsenceDays'])
df['LClass'] = le.fit_transform(df['Class'])
df.head(1)


# In[28]:


df=df.drop(["gender"],axis=1)
df=df.drop(["NationalITy"],axis=1)
df=df.drop(["PlaceofBirth"],axis=1)
df=df.drop(["StageID"],axis=1)
df=df.drop(["GradeID"],axis=1)
df=df.drop(["SectionID"],axis=1)
df=df.drop(["Topic"],axis=1)
df=df.drop(["Semester"],axis=1)
df=df.drop(["Relation"],axis=1)
df=df.drop(["ParentAnsweringSurvey"],axis=1)
df=df.drop(["StudentAbsenceDays"],axis=1)
df=df.drop(["ParentschoolSatisfaction"],axis=1)
df=df.drop(["Class"],axis=1)
df.head()


# In[29]:


df.to_csv('data.csv')


# # Univariate Selection

# In[30]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[31]:


x=df.iloc[:,df.columns !='LClass']
y=df.iloc[:,df.columns =='LClass']


# In[32]:


x.head()


# In[33]:


y.head()


# In[34]:


bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)


# In[35]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
featureScores.nlargest(10,'Score')  


# In[36]:


corr = df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# # Classification

# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


# In[39]:


x_train.head()


# In[40]:


y_train.head()


# # Models

# ## LogisticRegression

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


lr=LogisticRegression()


# In[43]:


lr.fit(x_train,y_train)


# In[44]:


predict1=lr.predict(x_test)
#predict1


# In[45]:


model1=accuracy_score(y_test,predict1)
print(model1)


# ## SVM

# In[46]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')


# In[47]:


svclassifier.fit(x_train,y_train)


# In[48]:


predict2=svclassifier.predict(x_test)


# In[49]:


model2=accuracy_score(y_test,predict2)
print(model1)


# # Neural_Network

# In[50]:


from sklearn.neural_network import MLPClassifier


# In[51]:


nn = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)


# In[52]:


nn.fit(x_train,y_train)


# In[53]:


predict3=nn.predict(x_test)


# In[54]:


model3=accuracy_score(y_test,predict3)
print(model3)


# In[55]:


import matplotlib.pyplot as plt; plt.rcdefaults()

objects = (' LogisticRegression','SVM','Neural_Network')
y_pos = np.arange(len(objects))
performance = [model1,model2,model3]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('LogisticRegression vs SVM vs NeighborsClassifier vs RandomForestClassifier')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




