# Machine_leanrning_coursework
Machine learning coursework part 2

# How to run
1. Clone this repository
2. All the files will get saved into a directory named `Machine_learning_coursework`
3. Open the terminal.
4. Go to directory `cd Machine_learning_coursework`
5. Install the required depedencies `$ pip install -r requirements.txt`
6. Run the code `python Part2.py` 



# Overview
This code `part2.py` performs sentiment classification on the given datasets. 
The steps are explained in the description section

# Files 
The repository contains 6 text files and a file (`requirements.txt`) enlisting dependencies.

# Dataset 
The 6 other text files are based on dataset containing 25,000 reviews (positive and negative) split into train, development
and test sets. The overall distribution of labels is roughly balanced.


# Description
This code takes these six text input files.  
Two files each appended into 3 datasets for the purposes of training, development and testing.
Reviews in each dataset are cleansed, lemmatized and tokenized.
They are converted into vectors by defining these tokens into four different features.
A Naive bayes classifier is trained using training set, which is tuned using developement set and tested upon the test set.
Chi-square reduction is used to further reduce the redundant features, imporve the accuracy and avoid over fitting.

