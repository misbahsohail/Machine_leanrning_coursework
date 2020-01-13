# Machine_leanrning_coursework
Machine learning coursework part 2

# How to run
1. Clone this repository
2. Go to directory `cd Machine_learning_coursework`
3. Download the required packages `$ pip install -r requirements.txt`
4. Run the code `python Part2.py`
5. If the code does not run, there might be some issue in the local machine

# Description
This code takes six file inputs.  
1. imdb_train_pos and imdb_train_neg - two files containing positive and negative reviews respectively for training purpose
2. imdb_dev_pos and imdb_dev_neg - two files containing positive and negative reviews respectively for developement purpose
3. imdb_test_pos and imdb_test_neg - two files containing positive and negative reviews respectively for testing purpose
Two files of same purpose files are appended into 3 single datasets
All reviews in each dataset are cleansed, lemmatized and tokenized
They are coverted into vectors by defining these tokens into four different features
A Naive bayes classifier is trained using vectorized training sets, further improved using vectorized developement sets and tested upon the vectorized test set
Chi-square reduction is used to further reduce the redundant features to further imporve the accuracy and to avoid over fitting

