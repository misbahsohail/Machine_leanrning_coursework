# Machine_leanrning_coursework
Machine learning coursework part 2

# How to run
1. Clone this repository
2. Go to directory `cd Machine_learning_coursework`
3. Install the required depedencies `$ pip install -r requirements.txt`
4. Run the code `python Part2.py`


# Description
The repository contains 6 text files and a file (`requirements.txt`) enlisting dependencies.
2 files containg positive and negative reviews are used for training and develepment purposes.
This code takes six input files.  
two files each appended into 3 datasets for the purposes of training development and testing.
Reviews in each dataset are cleansed, lemmatized and tokenized.
They are converted into vectors by defining these tokens into four different features.
A Naive bayes classifier is trained using training set, which is tuned using developement set and tested upon the test set.
Chi-square reduction is used to further reduce the redundant features, imporve the accuracy and avoid over fitting

