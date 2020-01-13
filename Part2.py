#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:51:45 2019

@author: c1950696
"""

def get_list_tokens(string):
    string=re.sub('[^A-Za-z]', ' ', string)
    no_of_tokens=0
    sentence_split=nltk.tokenize.sent_tokenize(string)
    list_tokens=[]
    for sentence in sentence_split:
        list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            token=token.lower()
            if token in stopwords: continue
            if token not in stopwords:
                if token not in list_tokens:
                    list_tokens.append(lemmatizer.lemmatize(token).lower())
                    no_of_tokens=no_of_tokens+1
                    #print('not stop word: ',token)
    return list_tokens,no_of_tokens


#importing libraries
import numpy as np

import nltk
import re



#nltk.download('punkt') # If needed
#nltk.download('wordnet') # If needed

# Importing the dataset
file1 = open("imdb_train_neg.txt", "r")
negfile=file1.read()
negfile_reviews=negfile.split('\n')

file2 = open("imdb_train_pos.txt", "r")
posfile=file2.read()
posfile_reviews=posfile.split('\n')

file3 = open("imdb_dev_neg.txt", "r")
dev_neg_file=file3.read()
dev_neg_reviews=dev_neg_file.split('\n')

file4 = open("imdb_dev_pos.txt", "r")
dev_pos_file=file4.read()
dev_pos_reviews=dev_pos_file.split('\n')

file5 = open("imdb_test_neg.txt", "r")
test_neg_file=file5.read()
test_neg_reviews=test_neg_file.split('\n')

file6 = open("imdb_test_pos.txt", "r")
test_pos_file=file6.read()
test_pos_reviews=test_pos_file.split('\n')






#Appending positive and negative reviews in a single set
dataset_full=[]
for pos_review in posfile_reviews:
  dataset_full.append((pos_review,1))
for neg_review in negfile_reviews:
  dataset_full.append((neg_review,0))

dataset_dev_full=[]
for pos_review in dev_pos_reviews:
  dataset_dev_full.append((pos_review,1))
for neg_review in dev_neg_reviews:
  dataset_dev_full.append((neg_review,0))

dataset_test_full=[]
for pos_review in test_pos_reviews:
  dataset_test_full.append((pos_review,1))
for neg_review in test_neg_reviews:
  dataset_test_full.append((neg_review,0)) 
  
 

#cleaning the reviews
nltk.download('stopwords')
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("!")
stopwords.add("?")
stopwords.add(";")
stopwords.add('"')
stopwords.add(":")
stopwords.add("(")
stopwords.add(")")
stopwords.add("who")
stopwords.add("where")
stopwords.add("when")
stopwords.add("how")
stopwords.add("the")







#-------------------------------------------------------------------------------------------------------------------------------#
#for Feature 1 and 2


X_train_F2=np.zeros((len(dataset_full), 1),dtype=int)
Y_train=[]
processed_reviews=[]
list_tokens=[]
for review_index in range(len(dataset_full)):   
    list_tokens,no_of_tokens=get_list_tokens(dataset_full[review_index][0])
    processed_reviews.append(list_tokens)
    processed_reviews[review_index]=' '.join(processed_reviews[review_index])
    Y_train.append(dataset_full[review_index][1])
 
    X_train_F2[review_index][0]=no_of_tokens
   

X_dev_F2=np.zeros((len(dataset_dev_full), 1),dtype=int)
list_tokens=[]
processed_dev_reviews=[] 
Y_dev_gold=[]
for review_index in range(len(dataset_dev_full)):   
    list_tokens,no_of_tokens=get_list_tokens(dataset_dev_full[review_index][0])
    processed_dev_reviews.append(list_tokens)
    processed_dev_reviews[review_index]=' '.join(processed_dev_reviews[review_index])
    Y_dev_gold.append(dataset_dev_full[review_index][1])   
    
    X_dev_F2[review_index][0]=no_of_tokens
    #no_of_tokens_dev[review_index][1]=dataset_full[review_index][1]

list_tokens=[]
X_test_F2=np.zeros((len(dataset_test_full), 1),dtype=int)
processed_test_reviews=[] 
Y_test_gold=[]
for review_index in range(len(dataset_test_full)):   
    list_tokens,no_of_tokens=get_list_tokens(dataset_test_full[review_index][0])
    processed_test_reviews.append(list_tokens)
    processed_test_reviews[review_index]=' '.join( processed_test_reviews[review_index])
    Y_test_gold.append(dataset_test_full[review_index][1])
    
    X_test_F2[review_index][0]=no_of_tokens
#----------------------------------------------------------------------------------------------------------------------------#
#creating a feature matrix for Feature 1
from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf_vect = TfidfVectorizer(min_df=0.01, max_df=1.0) #82.58696521391443 50.35985605757697
Tfidf_vect.fit(processed_reviews)

X_train_F1 = Tfidf_vect.transform(processed_reviews).toarray()
X_dev_F1 = Tfidf_vect.transform(processed_dev_reviews).toarray()
X_test_F1 = Tfidf_vect.transform(processed_test_reviews).toarray()

#-----------------------------------------------------------------------------------------------#
#creating a matrix to count the frequency of each word in each review


nltk.download('sentiwordnet')

from sklearn.feature_extraction.text import CountVectorizer


#Feature 3

CVvectorizer_feature3=CountVectorizer(min_df=0.01, max_df=1.0)

X_Train_array_F3=CVvectorizer_feature3.fit_transform(processed_reviews).toarray()  
X_Dev_array_F3=CVvectorizer_feature3.transform(processed_dev_reviews).toarray()
X_Test_array_F3=CVvectorizer_feature3.transform(processed_test_reviews).toarray()

vocab_of_total_words_F3=CVvectorizer_feature3.get_feature_names()

#Feature 4

CVvectorizer_feature4=CountVectorizer(max_features=6000)
X_Train_array_F4=CVvectorizer_feature4.fit_transform(processed_reviews).toarray()  
X_Dev_array_F4=CVvectorizer_feature4.transform(processed_dev_reviews).toarray()
X_Test_array_F4=CVvectorizer_feature4.transform(processed_test_reviews).toarray()

vocab_of_total_words_F4=CVvectorizer_feature4.get_feature_names()


#---------------------------------------------------------------------------------------------------------------#
#For Feature  3
from nltk.corpus import sentiwordnet as swn_word

def third_feature(vocab_of_total_words,X_set):  #3rd feature
 words_polarity_standard=[]
 for word in vocab_of_total_words: #for each word in the list vocabulary 
     try: #putting it in the try block because there can be a case when the word is not present in the sentilist
        synset=list(swn_word.senti_synsets(word))
        common_meaning=synset[0]
        
        if common_meaning.pos_score()>common_meaning.neg_score():
            
            #weight=common_meaning.pos_score()
            weight=1
        
        elif common_meaning.pos_score()<common_meaning.neg_score():
            
            #weight=-common_meaning.neg_score()
            weight=2
        else:
            weight=0
     except:
        
        weight=0
    
     words_polarity_standard.append(weight)
     
 
 Feature3=[] 

 for row in X_set:  
    


    words_polarity_standard_array=np.array(words_polarity_standard)
   
    
    
    weights_of_all_words_in_review=np.multiply(row,words_polarity_standard_array)
    
    
        
    Feature3.append(weights_of_all_words_in_review)  #this gives a list
    
    
 Feature3=np.vstack(Feature3)
 
 return Feature3
   
X_train_F3=third_feature(vocab_of_total_words_F3,X_Train_array_F3)

X_dev_F3=third_feature(vocab_of_total_words_F3,X_Dev_array_F3)

X_test_F3=third_feature(vocab_of_total_words_F3,X_Test_array_F3)


#----------------------------------------------------------------------------------------------------------------#
#For feature 4
def forth_feature(vocab_of_total_words,X_set): #4th feature
 words_polarity_standard=[]
 for word in vocab_of_total_words: #for each word in the list vocabulary 
     try: #putting it in the try block because there can be a case when the word is not present in the sentilist
        synset=list(swn_word.senti_synsets(word))
        common_meaning=synset[0]
        
        if common_meaning.pos_score()>common_meaning.neg_score():
            
            weight=common_meaning.pos_score()
        
        elif common_meaning.pos_score()<common_meaning.neg_score():
            
            weight=-common_meaning.neg_score()
            
        else:
            weight=0
     except:
        
        weight=0
    
     words_polarity_standard.append(weight)
     
 no_of_pos_and_neg=np.zeros((len(X_set), 2),dtype=int)
   
 count=0

 for row in X_set:  


    words_polarity_standard_array=np.array(words_polarity_standard)
   
    positive_word_count=0
    negative_word_count=0
    
    weights_of_all_words_in_review=np.multiply(row,words_polarity_standard_array)
    
    for a_word in weights_of_all_words_in_review:
        
        if a_word>0:
            positive_word_count=positive_word_count+1
        if a_word<0:
            negative_word_count=negative_word_count+1    
    
    no_of_pos_and_neg[count][0]=positive_word_count
    no_of_pos_and_neg[count][1]=negative_word_count
    
    count=count+1

 return no_of_pos_and_neg
  
X_train_F4=forth_feature(vocab_of_total_words_F4,X_Train_array_F4)

X_dev_F4=forth_feature(vocab_of_total_words_F4,X_Dev_array_F4)

X_test_F4=forth_feature(vocab_of_total_words_F4,X_Test_array_F4)



#-----------------------------------------------------------------------------------------------#
#Testing each feature seperately

from sklearn.naive_bayes import GaussianNB

Naive_F1 = GaussianNB()  #TF-IDF Vector
Naive_F1.fit(X_train_F1,Y_train)
predictions_dev_NB_F1 = Naive_F1.predict(X_dev_F1)
#predictions_test_NB_F1 = Naive_F1.predict(X_test_F1)


Naive_F3 = GaussianNB() #Word polarity containing in each review 
Naive_F3.fit(X_train_F3,Y_train)
predictions_dev_NB_F3 = Naive_F3.predict(X_dev_F3)
#predictions_test_NB_F3 = Naive_F3.predict(X_test_F3)  

Naive_F4 = GaussianNB() #Total Number of positive and negative words
Naive_F4.fit(X_train_F4,Y_train)
predictions_dev_NB_F4 = Naive_F4.predict(X_dev_F4)
#predictions_test_NB_F4 = Naive_F4.predict(X_test_F4)  


# Use accuracy_score function to get the accuracy


#from sklearn import metrics
#print("Naive Bayes Accuracy Score using TF-IDF -> ",metrics.accuracy_score(predictions_dev_NB_F1, Y_dev_gold)*100)
#print("Naive Bayes Accuracy Score using words polarityy -> ",metrics.accuracy_score(predictions_dev_NB_F3, Y_dev_gold)*100)
#print("Naive Bayes Accuracy Score using number of pos & neg words -> ",metrics.accuracy_score(predictions_dev_NB_F4, Y_dev_gold)*100)
#-------------------------------------------------------------------------------------------------------------------#
#Removing redundant features from each feature using Chi square

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

F3_new_features=SelectKBest(chi2, k=300).fit(X_train_F3, Y_train)
X_train_F3_new = F3_new_features.transform(X_train_F3)
#X_dev_F3_new = F3_new_features.transform(X_dev_F3)
X_test_F3_new = F3_new_features.transform(X_test_F3)

#X_train_new = SelectKBest(chi2, k=500).fit_transform(X_train, Y_train)
#print ("Size original training matrix: "+str(X_train_F3.shape))
#print ("Size new training matrix: "+str(X_train_F3_new.shape))


Naive_F3_new = GaussianNB()
Naive_F3_new.fit(X_train_F3_new,Y_train)

#predictions_dev_NB_F3_new = Naive_F3_new.predict(X_dev_F3_new)
predictions_test_NB_F3_new = Naive_F3_new.predict(X_test_F3_new)

#print("Naive Bayes Accuracy Score using words polarityy -> ",metrics.accuracy_score(predictions_dev_NB_F3_new, Y_dev_gold)*100)
#print("Naive Bayes Accuracy Score using words polarityy -> ",metrics.accuracy_score(predictions_test_NB_F3_new, Y_test_gold)*100)

 
F1_new_features=SelectKBest(chi2, k=1000).fit(X_train_F1, Y_train)
X_train_F1_new = F1_new_features.transform(X_train_F1)
#X_dev_F1_new = F1_new_features.transform(X_dev_F1)
X_test_F1_new = F1_new_features.transform(X_test_F1)

#X_train_new = SelectKBest(chi2, k=500).fit_transform(X_train, Y_train)
#print ("Size original training matrix: "+str(X_train_F1.shape))
#print ("Size new training matrix: "+str(X_train_F1_new.shape))


Naive_F1_new = GaussianNB()
Naive_F1_new.fit(X_train_F1_new,Y_train)

#predictions_dev_NB_F1_new =Naive_F1_new.predict(X_dev_F1_new)
predictions_test_NB_F1_new =Naive_F1_new.predict(X_test_F1_new)

#print("Naive Bayes Accuracy Score using TF-IDF -> ",metrics.accuracy_score(predictions_dev_NB_F1_new, Y_dev_gold)*100)
#print("Naive Bayes Accuracy Score using TF-IDF -> ",metrics.accuracy_score(predictions_test_NB_F1_new, Y_test_gold)*100)



#------------------------------------------------------------------------------------------------------------------#
#Concatinating all features together:


X_final_features=np.concatenate((X_train_F1_new,X_train_F2,X_train_F3_new,X_train_F4),axis=1)

#final_dev_data=np.concatenate((X_dev_F1_new,X_dev_F2, X_dev_F3_new, X_dev_F4),axis=1)

X_final_test=np.concatenate((X_test_F1_new,X_test_F2, X_test_F3_new, X_test_F4),axis=1)

Naive_Final_classifier = GaussianNB()
Naive_Final_classifier.fit(X_final_features,Y_train)
#final_predictions_dev = Naive_Final_classifier.predict(final_dev_data)
final_predictions_test = Naive_Final_classifier.predict(X_final_test)



Final_new_features=SelectKBest(chi2, k=850).fit(X_final_features, Y_train)
X_train_Final_new = Final_new_features.transform(X_final_features)
#X_Dev_Final_new = Final_new_features.transform(final_dev_data)
X_Test_Final_new = Final_new_features.transform(X_final_test)

Naive_Final_classifier_new = GaussianNB()
Naive_Final_classifier_new.fit(X_train_Final_new,Y_train)
#final_predictions_dev_new = Naive_Final_classifier_new.predict(X_Dev_Final_new)
final_predictions_test_new = Naive_Final_classifier_new.predict(X_Test_Final_new)


#-------------------------------------------------------------------------------------------------------------#
#Final Results

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score



precision=precision_score(Y_test_gold, final_predictions_test_new, average='macro')
recall=recall_score(Y_test_gold, final_predictions_test_new, average='macro')
f1=f1_score(Y_test_gold, final_predictions_test_new, average='macro')
accuracy=accuracy_score(Y_test_gold, final_predictions_test_new)


print ("Precision: "+str(precision))
print ("Recall: "+ str(recall))
print ("F1-Score: "+str(f1))
print ("Accuracy: "+str(accuracy))

from sklearn.metrics import confusion_matrix

print (confusion_matrix(Y_test_gold, final_predictions_test_new))




