#install catboost
!pip install catboost

#import data
import pandas as pd
from google.colab import files
uploaded = files.upload()
import io
df = pd.read_csv(io.BytesIO(uploaded['Data.csv']), sep = ";")
df

#splitting the sample into predictors and response
X = df.loc[:, df.columns != 'Y']
Y = df.loc[:, df.columns == 'Y']

#selection of predictors
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100).fit(X, y), prefit=True, threshold=0.02)
print(selector.get_support())
Xs = selector.transform(X)

names = list(df.columns)
dict(zip(names, selector.get_support()))


from sklearn.model_selection import KFold, cross_val_predict, train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score, precision_score, recall_score
from catboost import CatBoostClassifier
import pickle

#initialization CatBoosting model
classifier=CatBoostClassifier(iterations=300,
                           depth=3,
                           learning_rate=0.1/3,
                           loss_function='Logloss',
                           verbose=False,
                           class_weights={0: 1, 1: 1.7})

#initialization K-fold validation                         
cv = KFold(n_splits=10, shuffle=True)
prediction = []
prob_prediction = []
labels = []

#training model
for train_index, test_index in cv.split(Xs):
    Xs_train, Xs_test = Xs[train_index], Xs[test_index]
    ys_train, ys_test = y.iloc[train_index], y.iloc[test_index]

    clf = classifier.fit(Xs_train, ys_train)

    prediction.append(clf.predict(Xs_test))
    prob_prediction.append(clf.predict_proba(Xs_test)[:, 1])
    labels.append(ys_test)

prediction = np.concatenate(prediction)
labels = np.concatenate(labels)
prob_prediction = np.concatenate(prob_prediction)
prediction = (prob_prediction > 0.5).astype(int)  
print("Accuracy:", accuracy_score(prediction, labels))
print("Precision:", precision_score(prediction, labels))
print("Recall:", recall_score(prediction, labels))
print("ROC-AUC:", roc_auc_score(labels, prob_prediction))

#save the model
from joblib import dump, load
dump(clf, 'catboost_clf.joblib') 

#test the model on whole data
loaded_model = load('catboost_clf.joblib')

from sklearn.metrics import confusion_matrix
expected=y
predicted=loaded_model.predict(Xs)
conf_matrix = confusion_matrix(expected, predicted)
print(conf_matrix)
