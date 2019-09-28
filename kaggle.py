import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
# print(train.info())
# print(test.info())
# 发现Embarked缺失
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']
# print(X_train['Embarked'].value_counts())
# print(X_test['Embarked'].value_counts())
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
print(X_train.info())
print(X_test.info())
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
# print(dict_vec.feature_names_)
X_test = dict_vec.transform(X_test.to_dict(orient='record'))
rfc = RandomForestClassifier()
xgbc = XGBClassifier()

rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('data/rfc_submission.csv', index=False)

xgbc.fit(X_train,y_train)
XGBClassifier(base_score=0.5,colsample_bylevel=1,colsample_bytree=1,
              gamma=0,learning_rate=0.1,max_delta_step=0,max_depth=3,
              min_child_weight=1,missing=None,n_estimators=100,nthread=1,
              objective='binary:logistic',reg_alpha=0,reg_lambda=1,
              scale_pos_weight=1,seed=0,silent=True,subsample=1
              )
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('data/xgbc_submission.csv',index=False)

print(cross_val_score(rfc,X_train,y_train,cv=5).mean())
print(cross_val_score(xgbc,X_train,y_train,cv=5).mean())
