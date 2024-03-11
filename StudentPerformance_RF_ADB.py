#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


df=pd.read_csv('xAPI-Edu-Data.csv')


# # Data Analysis

# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.dropna() 


# In[ ]:


# Check if missing values exist or not
df.isnull().sum()


# # EDA

# In[ ]:


#plt.subplot(1,2,1)
sns.countplot(x="gender", order=['F','M'], data=df, palette="Set1")
plt.show()


# In[ ]:


#plt.subplot(1,2,2)
sns.countplot(x="gender", order=['F','M'], hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[ ]:


#plt.subplot(1,2,1)
df['NationalITy'].value_counts(normalize=True).plot(kind='bar')
plt.show()


# In[ ]:


#plt.subplot(1,2,2)
df['PlaceofBirth'].value_counts(normalize=True).plot(kind='bar')
plt.show()


# In[ ]:


#plt.subplot(1,2,1)
sns.countplot(y="NationalITy", data=df, palette="muted")
plt.show()


# In[ ]:


#plt.subplot(1,2,2)
sns.countplot(y="NationalITy", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[ ]:


sns.countplot(x="Relation", order=['Mum','Father'], data=df, palette="Set1")
plt.show()


# In[ ]:


sns.countplot(x="Relation", order=['Mum','Father'], hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[ ]:


sns.countplot(x="StageID", data=df, palette="muted")
plt.show()


# In[ ]:


sns.countplot(x="StageID", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[ ]:


sns.countplot(x="GradeID", data=df, palette="muted")
plt.show()


# In[ ]:


sns.countplot(x="GradeID", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[ ]:


plt.subplot(1,2,1)
sns.countplot(x="SectionID", order=['A','B','C'], data=df, palette="muted")

plt.subplot(1,2,2)
sns.countplot(x="SectionID", order=['A','B','C'], hue="Class", hue_order=['L','M','H'], data=df, palette="muted")

plt.show()


# In[ ]:


plt.subplot(1,2,1)
sns.countplot(y="Topic", data=df, palette="muted")

plt.subplot(1,2,2)
sns.countplot(y="Topic", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")

plt.show()


# In[ ]:


#plt.subplot(1,2,1)
sns.countplot(x="ParentschoolSatisfaction", data=df, palette="muted")
plt.show()


# In[ ]:


#plt.subplot(1,2,2)
sns.countplot(x="ParentschoolSatisfaction", hue="Class", hue_order=['L','M','H'], data=df, palette="muted")
plt.show()


# In[ ]:


# Checking dataset 


# In[ ]:


plt.figure(figsize=(8, 8))
sns.countplot('Class', data=df)
plt.title('Balanced Classes')
plt.show()


# # Pre-processing

# In[ ]:


from sklearn import preprocessing
le=preprocessing.LabelEncoder()


# In[ ]:


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


# In[ ]:


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


# In[ ]:


df.to_csv('data.csv')


# # Univariate Selection

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


x=df.iloc[:,df.columns !='LClass']
y=df.iloc[:,df.columns =='LClass']


# In[ ]:


x.head()


# In[ ]:


y.head()


# In[ ]:


bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)


# In[ ]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
featureScores.nlargest(10,'Score')  


# In[ ]:


corr = df.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# # Classification

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# # Models

# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf=RandomForestClassifier()


# In[ ]:


rf.fit(x_train,y_train)


# In[ ]:


predic=rf.predict(x_test)
acc1=accuracy_score(predic,y_test)
acc1


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=50, learning_rate=1)


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


predic=rf.predict(x_test)
acc2=accuracy_score(predic,y_test)
acc2


# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()

objects = ('rf','model ')
y_pos = np.arange(len(objects))
performance = [acc1,acc2]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Random Forest vs Adaboost')

plt.show()


# In[ ]:




