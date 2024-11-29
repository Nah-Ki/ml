from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print("Length of the twenty_train--------->", len(twenty_train.data))

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print('dim=', X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
accuracy = np.mean(predicted == twenty_test.target)
print("Predicted Accuracy =", accuracy)

from sklearn import metrics
print("Accuracy=", metrics.accuracy_score(twenty_test.target, predicted))
print("Precision=", metrics.precision_score(twenty_test.target, predicted, average=None))
print("Recall=", metrics.recall_score(twenty_test.target, predicted, average=None))
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
