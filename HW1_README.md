# Sentiment Analysis of BLM movement Tweets
## Dataset Information

We use and compare various different methods for sentiment analysis on tweets (a binary classification problem). The data set consisting of 8435 tweets (the training(6747 tweets) and test (1688) tweet data sets)to build an opinion classifier, or a method that classifies a tweet as positive or negative with respect to the #BlackLivesMatter movement.

## Requirements
There are some general library requirements for the project and some which are specific to individual methods. The general requirements are as follows.  

* `numpy`
* `scikit-learn`
* `scipy`
* `nltk`
* `Pandas`
* `string`
* `emoji`
* `regex`

**Note**: It is recommended to use Anaconda distribution of Python.

## Usage

### Preprocessing 
This notebook contains the code to preprocess tweets and create a bag of words for the train and test data.

### Classifier
* `naive bayes`
* `multinomial`
* `desicion tree`
* `knn`

### Evaluation
* `accuracy`
* `precesion,recall,f1-score`
* `roc_auc_score`



