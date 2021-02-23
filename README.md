README.md
Sentiment Analysis for Event-Driven Stock Prediction
Use natural-language processing (NLP) to predict stock price movement based on Reuters News

Website
You are welcome to visit our website: GolenRocks.me. The main purpose of this project is to build the connection between Bayesian DNN and stock price prediction based on News headline.

Methodology
Data Collection and Preprocessing

1.1 crawl a ticker list to obtain the details of public companies

1.2 crawl news from Reuters using BeautifulSoup

1.3 crawl prices using urllib

Feature Engineering (Tokenization)

2.1 Unify word format: unify tense, singular & plural, remove punctuations & stop words

2.2 Implement one-hot encoding

2.3 Pad word sequence (essentially a matrix) to keep the same dimension

Train a set of Bayesian Convolutional Neural Networks using Stochastic Gradient Langevin Dynamics to obtain more robustness

Use thinning models to predict future news

Requirement
Python 3
PyTorch > 0.4
numpy
NLTK
Crawler tools
pip3 install lxml
pip3 install bs4
pip3 install urllib
Usage
Note: If you don't want to take time to crawl data and train the model, you can also directly go to step 4.

1. Data collection
1.1 Download the ticker list from NASDAQ
$ ./crawler/all_tickers.py 20  # keep the top e.g. 20% marketcap companies
1.2 Use BeautifulSoup to crawl news headlines from Reuters
Note: you may need over one month to fetch the news you want.

Suppose we find a piece of news about COO Lu Qi Resignation on May.18, 2018 at reuters.com



We can use the following script to crawl it and format it to our local file

$ ./crawler/reuters.py # we can relate the news with company and date, this is more precise than Bloomberg News


By brute-force iterating company tickers and dates, we can get the dataset with roughly 400,000 news in the end. Since a company may have multiple news in a single day, the current version will only use topStory news to train our models and ignore the others.

1.3 Use urllib to crawl historical stock prices
Improvement here, use normalized return [5] over S&P 500 instead of return.

$ ./crawler/yahoo_finance.py # generate raw data: stockPrices_raw.json, containing open, close, ..., adjClose
$ ./create_label.py # use raw price data to generate stockReturns.json
2. Feature engineering (Tokenization)
Unify the word format, project word to a word vector, so every sentence results in a matrix.

Detail about unifying word format are: lower case, remove punctuation, get rid of stop words, unify tense and singular & plural.

$ ./tokenize_news.py
3. Train a Bayesian ConvNet to predict the stock price movement.
Type the following to train a set of robust Bayesian models.

$ ./main.py -epochs 500 -static False
4. Prediction and analysis
Let's show one example how the thinning models react to Baidu Lu Qi's resignation

$ ./main.py -predict "Top executive behind Baidu's artificial intelligence drive steps aside"
>> Sell
The prediction makes sense, let's find another one.

Eli Lilly and Co (LLY.N)
FRI, JUN 1 2018
UPDATE 2-Lilly gets U.S. nod for arthritis drug, sets price well below rivals
* Drug priced at $25,000/year, 60 pct lower to AbbVie's Humira
$ ./main.py -predict "UPDATE 2-Lilly gets U.S. nod for arthritis drug  sets price well below rivals"
>> Sell
Lowering down drug prices looks helpful to gain market share in the business, however, they didn't mention too much about the updates of technology, we are inclined to regard it as the virulent price competition, which does not help to the company earnings. Thus it is not a bad decision to sell Eli Lilly stocks.

Next, let's see what the buy options look like:

Alphabet Inc (GOOG.O)
WED, MAY 30 2018
Google launches the second app in China, woos top smartphone market
* BEIJING Alphabet Inc's Google has launched a file managing tool in several Chinese app stores as it 
* looks for fresh inroads into the world's biggest smartphone market, where most of the internet 
* giant's top products remain banned.
$ ./main.py -predict "Google launches the second app in China  woos top smartphone market"
>> Strong Buy
By now, you have basically understood how the models work, let's use backtesting to examine the performance on the news in the past two weeks.

$ ./main.py -eval True
>> Testing    - loss: 0.6761  acc: 58.07%(41.8/72.0) 83.50%(3.3/3.9) 100.00%(0.0/0.0) 0.00%(0.0/0.0) 
Note: the predictions are averaged, which explains why we have float numbers. From left to right, the predictions become more and more confident. 58% is actually much higher than my expectation, I believe when tested on a longer time horizon, the performance gets worse. However, as long as the predictions are better than random guesses (50%), you can't lose money betting on a favorable game (assume no trading cost and liquidity issue).

5. Future works
This is a very rough work. A better label should be based on the comparison of stock price changes between the company and the corresponding industry, instead of the S&P 500, which is in spririt similar to hedging.

By Tim Loughran and Bill McDonald, some words have strong indications of positive and negative effects in finance, e.g. company merger and acquisition. Therefore we need to dig into these words to find more information. In addition, detailed analysis and comparison in each industry are also useful.

Another simple but interesting example can be found in Financial Sentiment Analysis part1, part2.

Since a comprehensive stopword list is quite helpful in improving the prediction power, you are very welcome to build a better stopword list and share it.

References:
Yoon Kim, Convolutional Neural Networks for Sentence Classification, EMNLP, 2014
J Pennington, R Socher, CD Manning, GloVe: Global Vectors for Word Representation, EMNLP, 2014
Max Welling, Yee Whye Teh, Bayesian Learning via Stochastic Gradient Langevin Dynamics, ICML, 2011
Tim Loughran and Bill McDonald, 2011, “When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks,” Journal of Finance, 66:1, 35-65.
H Lee, etc, On the Importance of Text Analysis for Stock Price Prediction, LREC, 2014
Xiao Ding, Deep Learning for Event-Driven Stock Prediction, IJCAI2015
IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW
Keras predict sentiment-movie-reviews using deep learning
Keras sequence-classification-lstm-recurrent-neural-networks
tf-idf + t-sne
Implementation of CNN in sequence classification
Getting Started with Word2Vec and GloVe in Python
PyTorch Implementation of Kim's Convolutional Neural Networks for Sentence Classification
