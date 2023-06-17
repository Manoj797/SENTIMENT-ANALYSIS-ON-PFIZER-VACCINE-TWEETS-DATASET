# SENTIMENT-ANALYSIS-ON-PFIZER-VACCINE-TWEETS-DATASET
Sentiment analysis models can be trained using a variety of machine learning models such as  logistic regression,svm,random forest with the help of pfizer vaccine tweets dataset in order  to classify the sentiment.

#ABSTRACT

Twitter sentiment analysis is a process of obtaining and examining the thoughts, attitudes, and 
feelings expressed by users on the social media platform Twitter. It involves categorizing tweets 
as positive, negative, or neutral based on the sentiment they convey. This analysis can provide 
valuable insights into consumer feedback, public opinion, brand reputation, and market trends. 
Natural language processing techniques, such as text preparation and feature extraction, are 
used in sentiment analysis. Machine learning models, including logistic regression, SVM, and 
random forest, can be trained using labeled datasets to classify sentiments. This study utilizes 
a dataset of Pfizer vaccine tweets to demonstrate the application of sentiment analysis models.


DATASET COLLECTION:


The vaccination_tweets is a dataset of 11314 tweets about the Pfizer vaccine. The data was 
collected using the Tweepy Python package to access the Twitter API. The dataset includes the 
following columns: id, user_name, user_location, user_description, user_created, 
user_followers, user_friends, user_favourites, user_verified, date, and text. In Pfizer Vaccine 
Tweets dataset here we make use of preprocessed dataset in which it contains 10543and 3 
columns such as text,polarity and sentiment.where sentiment is derived from the value which 
is obtained from an polarity and based on value from the polarity they have derived the 
sentiment column as an target class


PRE-PROCESSING METHODS:


Preprocessing data is a critical stage in the data mining process. It refers to the process of 
cleaning, converting, and combining data in order to prepare it for analysis. The purpose of 
data preparation is to enhance data quality and make it more suited for the specific data mining 
activity at hand.

 According to our dataset we have taken the partially preprocessed dataset i.e, the twitter 
texts are free from emojis, URLs, mentions, hashtags, non word character and nonwhitespace character, multiple places are removed. 
 And stop words also removed, the purpose of removing stopwords from text data during 
natural language processing tasks is to reduce noise and focus on the words that are 
more meaningful for analysis. And as we are doing analysis pf text only tweet text 
column was taken as the input. 
 And all words are converted to lower case. And stemming algorithm is used to covert 
texts into its stemmed version I.e., running, runs, ran will be converted to run. 
 Then the polarity is generated for each text using textblob library. The TextBlob class 
provides a simple API for common natural language processing tasks such as part-ofspeech tagging, noun phrase extraction, and sentiment analysis. It gives float value 
between -1.0 to 1.0. If the polarity is 0 then the sentiment is neutal, if greater than 0 
then positive and less than 0 then negative.
But the issue regarding to the dataset is class imbalance, to overcome that probelm we have 
used the concept called smote in order to balance the class. SMOTE's major purpose is to 
interpolate between existing minority class samples to develop synthetic instances for the 
minority class (the class with less samples). This approach aids in the balancing of class 
distributions and the improvement of machine learning model performance, particularly in 
circumstances when the minority class is underrepresented.Addition to this we checked 
whether there is an null values or not.one imbalancement column is target class by applying 
this smote algorithm we achived best accuracy and get all the classes balanced

 Logistic regression:

Logistic regression can be extended to handle multi-class classification problems through 
various techniques, such as One-vs-Rest (OvR) or softmax regression. Here we have used the 
OvR regression, since LogisticRegression class in scikit-learn internally uses the OvR strategy 
for multi-class classification.

The OvR approach is like a separate binary logistic regression model, it is trained for each 
class, considers one class as the positive class and the remaining classes as the negative class. 
During training, the probabilities of belonging to each class are calculated independently, and 
the class with the highest probability is predicted. 

We have divided the dataset into 20% of test data and 80% of the train data. Then train data 
has been fitted to the model. Then the model is evaluated for test data and confusion matrix is 
calculated. 

SVM:
A linear Support Vector Machine (SVM) model can be used for multi-class classification using 
one of two approaches: One-vs-Rest (OvR) or One-vs-One (OvO).
Here we have used OvR approach, it acts as a separate binary SVM classifier is trained for each 
class. During training, one class is treated as the positive class, and the rest of the classes are 
considered as the negative class. This process is repeated for each class, resulting in a binary 
classifier for each class. During prediction, the class label with the highest SVM score is 
assigned to the test instance.
Then we train the SVM classifier using the training data and evaluated for test data. We have 
divided the dataset into 20% of test data and 80% of the train data. Then train data has been 
fitted to the model. Then the model is evaluated for test data and confusion matrix is calculated.


 RandomForest:
 
Training Phase:

Initially Random sampling with replacement (bootstrapping) is used to create multiple training 
datasets (bootstrap samples) from the original dataset.For each bootstrap sample, a decision 
tree is trained on a random subset of features. The decision trees are grown by recursively 
partitioning the data based on feature thresholds that optimize a splitting, here we have used 
‘Gini’ criterion. Each decision tree is trained independently and can potentially overfit the 
training data. The number of decision trees to be grown is specified by the n_estimators 
parameter which is 200.

Prediction Phase:

During prediction, each decision tree in the random forest independently predicts the class label 
for a given input sample. For multi-class classification, the random forest combines the 
predictions of all decision trees to make a final prediction. Majority Voting, each decision tree 
"votes" for a class label, and the class label with the most votes across all trees is assigned as 
the final prediction. It is used for combining the predictions.

Result:

The dataset was trained using other models such as decision tree, naïve Bayesian but they gave 
accuracy less than 60%. The highest accuracy of 82% for this dataset is achieved by using 
Logistic regression. Linear SVM model gave the accuracy of 78% and random forest gave the 
accuracy of 72%. This accuracy was again evaluated by cross validation technique. By using 
cross validation technique, we got the accuracy of 78% for logistic regression, 74% for linear 
SVM and 69% for random forest by taking of all average
