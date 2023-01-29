
# coding: utf-8

# 
# # Assignment 3: Code for data modelling presentation

# In[1]:


#fill in your code here
#invoking modules


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix, ConfusionMatrixDisplay 
from imblearn.over_sampling import SMOTE 
import ssl
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 


# In[2]:


sms_path = 'A3_sms.csv'

sms_df = pd.read_csv(sms_path, header = 0, index_col = 0, sep = ',', decimal = '.', encoding = 'utf8')

print(sms_df)


print(sms_df.shape)
print(sms_df.dtypes)

# Look at break up between ham and spam sms texts
print(sms_df['spam'].value_counts())


# In[3]:


#preparation: 
sms_df['sms'] = sms_df['sms'].str.lower()
# Split the data frame between spam and ham text to check percentage of spam in dataset.

sms_ham = sms_df[sms_df['spam'] == False]
sms_spam = sms_df[sms_df['spam'] == True]

print(sms_spam['spam'].value_counts())
print('First 10 results of ham texts: \n', sms_ham[0:10])

print(sms_ham['spam'].value_counts())
print('First 10 results of ham texts: \n', sms_spam[0:10])



# In[4]:


# Data Preparation: - Ham
sms_ham
sms_ham = sms_ham.drop(columns = ['Unnamed: 3'], axis = 1)
print(sms_ham.dtypes)
sms_ham




# In[5]:


# Data Preparation: - Spam
# Data Preparation: - Ham
sms_spam
sms_spam = sms_spam.drop(columns = ['Unnamed: 3'], axis = 1)
print(sms_spam.dtypes)
sms_spam


# In[6]:


# Create Deep copy
sms_ham_prep = sms_ham.copy()
sms_spam_prep = sms_spam.copy()



# In[7]:


#Tokenisation
#convert to strings and copy data frame
sms_ham['sms'] = sms_ham['sms'].astype('str')
sms_spam['sms'] = sms_spam['sms'].astype('str')

## Deep Copies ##
sms_ham_prep['sms'] = sms_ham_prep['sms'].astype('str')
sms_spam_prep['sms'] = sms_spam_prep['sms'].astype('str')


## disabling SSl check to download the package "punkt"
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass 
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
# load tokens (words) 

nltk.download('punkt') 



# In[8]:


# Tokenization For ham:
#https://stackoverflow.com/questions/33098040/how-to-use-word-tokenize-in-data-frame reference
sms_ham_prep['sms'] = sms_ham_prep['sms'].apply(word_tokenize)
print('Head of sms ham: \n', sms_ham_prep['sms'].head()) 


# Tokenization For spam:
sms_spam_prep['sms'] = sms_spam_prep['sms'].apply(word_tokenize)
print('Head of sms spam: \n', sms_spam_prep['sms'].head()) 


# In[9]:


# Removing stopwords
# Removes insignificant words from dataset
nltk.download('stopwords') 
list_stopwords=stopwords.words('english') 
print(list_stopwords[0:10]) 

#For Ham
sms_ham_prep['sms'] = sms_ham_prep['sms'].apply(lambda x: [item for item in x if item not in list_stopwords])

# For Spam
sms_spam_prep['sms'] = sms_spam_prep['sms'].apply(lambda x: [item for item in x if item not in list_stopwords])

#print

print(sms_ham_prep['sms'].head())
print(sms_spam_prep['sms'].head())

#https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe


# In[10]:


# Data Exploration

# Determining whether data set is imablanced or not. 
len(sms_spam_prep)/len(sms_spam_prep+sms_ham_prep)

#Running the code we see that 13.16% of sms texts in data set is spam, showing that it is imbalanced.


# In[11]:


# Recombine ham and spam for countvectorize function

sms_test = sms_ham.append(sms_spam)

# Removes labeLS from dataframe before being passed into countvectorizer - an algorithm that transforms text into numerical data.
sms_test = sms_test.drop(columns = ['spam'], axis = 1)
sms_test







# In[12]:


# Feature Extraction from text
#https://www.quora.com/How-can-I-do-machine-learning-using-scikit-learn-for-a-CSV-file-that-has-many-text-columns
CountVec = CountVectorizer(lowercase=True,analyzer='word',stop_words='english') 
feature_vectors = CountVec.fit_transform(sms_test['sms'])

#show the extracted features from the dataset. These features are all the words found in the spam and ham texts.
CountVec.get_feature_names_out() 


# In[13]:


##### DATA MODELLING ###

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, [0] * len(sms_ham['spam']) + [1] * len(sms_spam['spam']), random_state = 0, test_size=0.2)



# In[14]:


#KNN and Hyperparameter tuning

knn = KNeighborsClassifier() 


# In[15]:


#Define the dictionary of hyperparameters (number of neighbours and p) to be evaluated using grid search. Finds the best parameters

hyperparameters = { 

        'n_neighbors': [1, 3, 5, 9, 11], 

        'p': [1, 2] 
}


# In[16]:


# Train and evaluate the KNN classifier
knn = GridSearchCV(knn, hyperparameters, scoring='accuracy') 

knn.fit(X_train, y_train) 

print('Best p:', knn.best_estimator_.get_params()['p']) 

print('Best n_neighbors:', knn.best_estimator_.get_params()['n_neighbors']) 


# In[17]:


# KNN Classifier Evaluation
results=pd.DataFrame() 

y_predicted = knn.predict(X_test)


# In[18]:


cm = confusion_matrix(y_test, y_predicted, labels=knn.classes_) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_) 

disp.plot() 


# In[19]:


#Accuracy
results.loc['knn','accuracy']=accuracy_score(y_test,y_predicted) 

print(results.loc['knn','accuracy']) 


# In[20]:


#Balanced Accuracy
results.loc['knn','balanced_accuracy']=balanced_accuracy_score(y_test,y_predicted) 

print(results.loc['knn','balanced_accuracy']) 


# In[21]:


#Training Time in Seconds
results.loc['knn','training_time']=knn.cv_results_['mean_fit_time'].mean() 

print(results.loc['knn','training_time']) 


# In[22]:


# Prediction Time in Seconds per text

results.loc['knn','prediction_time']=knn.cv_results_['mean_score_time'].mean()/len(y_test) 

print(results.loc['knn','prediction_time']) 


# In[23]:


## TRAIN AND EVALUATE KNN (SMOTE)
X_train_smote, y_train_smote =SMOTE().fit_resample(X_train, y_train) 


# In[24]:


#balances training set
y_train_smote.count(1)/len(y_train_smote)


# In[25]:


## TRAIN KNN (SMOTE) USING HYPERPARAMETER TUNING

knn.fit(X_train_smote, y_train_smote) 

print('Best p:', knn.best_estimator_.get_params()['p']) 

print('Best n_neighbors:', knn.best_estimator_.get_params()['n_neighbors']) 


# In[26]:


## KNN (SMOTE) CLASSIFIER EVALUATION
y_predicted = knn.predict(X_test) 


# In[27]:


# KNN(SMOTE) Confusion Matrix
cm = confusion_matrix(y_test, y_predicted, labels=knn.classes_) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_) 

disp.plot() 


# In[28]:


# Accuracy 
results.loc['knn_SMOTE','accuracy']=accuracy_score(y_test,y_predicted) 

print(results.loc['knn_SMOTE','accuracy']) 


# In[29]:


# Balanced Accuracy
results.loc['knn_SMOTE','balanced_accuracy']=balanced_accuracy_score(y_test,y_predicted) 


print(results.loc['knn_SMOTE','balanced_accuracy']) 


# In[30]:


# Training Time (in seconds)
results.loc['knn_SMOTE','training_time']=knn.cv_results_['mean_fit_time'].mean() 

print(results.loc['knn_SMOTE','training_time']) 


# In[31]:


# Prediction Time (in seconds) per text

results.loc['knn_SMOTE','prediction_time']=knn.cv_results_['mean_score_time'].mean()/len(y_test) 

print(results.loc['knn_SMOTE','prediction_time']) 


# In[32]:


###   DECISION TREE and HYPERPARAMETER TUNING   ###

#Define decision tree classifier

dt = DecisionTreeClassifier() 


# In[33]:


# Applying grid search to determing best parameters for Decision Tree

hyperparameters = { 

    'min_samples_split': [2,3,5], 

    'min_samples_leaf': [5, 10, 20, 50, 100], 

    'max_depth': [2, 3, 5, 10, 20] 

} 


# Setting grid search to determine best parameters for 'Accuracy'
dt = GridSearchCV(dt, hyperparameters, scoring='accuracy')


# In[34]:


# TRAIN AND EVALUATE DECISION TREE CLASSIFIER

dt.fit(X_train, y_train) 

print('Best max_depth:', dt.best_estimator_.get_params()['max_depth']) 

print('Best min_samples_leaf:', dt.best_estimator_.get_params()['min_samples_leaf']) 

print('Best criterion:', dt.best_estimator_.get_params()['criterion']) 


# In[35]:


# DECISION TREE CLASSIFIER EVALUATION
y_predicted = dt.predict(X_test)


# In[36]:


# Metrics for evaluation for Decision Tree

cm = confusion_matrix(y_test, y_predicted, labels=dt.classes_) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_) 

disp.plot() 


# In[37]:


# Accuracy

results.loc['dt','accuracy']=accuracy_score(y_test,y_predicted) 

print(results.loc['dt','accuracy']) 


# In[38]:


# Balanced Accuracy

results.loc['dt','balanced_accuracy']=balanced_accuracy_score(y_test,y_predicted) 

print(results.loc['dt','balanced_accuracy']) 


# In[39]:


# Training Time (in seconds)
results.loc['dt','training_time']=dt.cv_results_['mean_fit_time'].mean() 

print(results.loc['dt','training_time']) 


# In[40]:


# Prediction time (in seconds) per text

results.loc['dt','prediction_time']=dt.cv_results_['mean_score_time'].mean()/len(y_test) 

print(results.loc['dt','prediction_time']) 


# In[41]:


# TRAIN AND EVALUATE DECISION TREE USING SMOTE

dt.fit(X_train_smote, y_train_smote) 

print('Best max_depth:', dt.best_estimator_.get_params()['max_depth']) 

print('Best min_samples_leaf:', dt.best_estimator_.get_params()['min_samples_leaf']) 

print('Best criterion:', dt.best_estimator_.get_params()['criterion']) 


# In[42]:


# DECISION TREE (SMOTE) CLASSIFIER EVALUATION

y_predicted = dt.predict(X_test)  


# In[43]:


# Metrics for evaluation of Decision Tree (SMOTE)

cm = confusion_matrix(y_test, y_predicted, labels=dt.classes_) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_) 

disp.plot() 


# In[44]:


# Accuracy

results.loc['dt_SMOTE','accuracy']=accuracy_score(y_test,y_predicted) 

print(results.loc['dt_SMOTE','accuracy']) 


# In[45]:


# Balanced Accuracy
results.loc['dt_SMOTE','balanced_accuracy']=balanced_accuracy_score(y_test,y_predicted) 

print(results.loc['dt_SMOTE','balanced_accuracy']) 


# In[46]:


# Training Time (in seconds) per text
results.loc['dt_SMOTE','training_time']=dt.cv_results_['mean_fit_time'].mean() 

print(results.loc['dt_SMOTE','training_time']) 


# In[47]:


# Prediction Time (in seconds) per text

results.loc['dt_SMOTE','prediction_time']=dt.cv_results_['mean_score_time'].mean()/len(y_test) 

print(results.loc['dt_SMOTE','prediction_time']) 


# In[48]:


####       Results     ####

# Results of 4 implemented classifiers: KNN, Decision Trees and also using SMOTE on both models to balance the data set.

results


# In[49]:


# Accuracy and Balanced Accuracy

results[['accuracy','balanced_accuracy']].plot(kind="bar") 
plt.title('Accuracy and Balanced Accuracy of Classifiers')
plt.xlabel('Classifier Models') 
plt.ylabel('Percentage')
plt.show()


# In[50]:


# Training Time (in seconds)

results['training_time'].plot(kind="bar") 
plt.title('Training Time (in seconds)')
plt.xlabel('Classifier Models') 
plt.ylabel('Time')
plt.show()


# In[51]:


# Prediction Time (in seconds) per text

results['prediction_time'].plot(kind="bar") 
plt.title('Prediction Time (in seconds)')
plt.xlabel('Classifier Models') 
plt.ylabel('Time')
plt.show()


# In[52]:


# Most common words
#https://stackoverflow.com/questions/29903025/count-most-frequent-100-words-from-sentences-in-dataframe-pandas

from collections import Counter
common_hams = Counter(" ".join(sms_ham['sms']).split()).most_common(100)
common_spams = Counter(" ".join(sms_spam['sms']).split()).most_common(100)

print('most common hams are:' , common_hams)
print('most common spams are:', common_spams)


# In[ ]:




