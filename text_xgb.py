from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


train_df = pd.read_csv('train_set.csv', sep='\t')

tfidf = TfidfVectorizer(min_df=20, max_df=0.7, max_features=8000)
X = tfidf.fit_transform(train_df['text'])
y = train_df['label']

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

logistic_model = XGBClassifier(learning_rate=0.05,
              n_jobs=-1,
              n_estimators=400,           
              max_depth=10,               
              gamma=0.5,                                   
              subsample=0.8)
#logistic_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logistic_model.fit(train_X, train_y)

y_pred = logistic_model.predict(test_X)

from sklearn.metrics import f1_score
f1_score(test_y, y_pred, average='macro')



test_df = pd.read_csv('test_a.csv', sep='\t')

X_ = tfidf.transform(test_df['text'])
y_ = logistic_model.predict(X_)
result = pd.DataFrame(y_,columns=['label'])
result.to_csv('test1.csv',encoding='utf8',index=0)