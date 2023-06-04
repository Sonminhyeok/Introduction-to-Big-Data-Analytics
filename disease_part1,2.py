
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import catboost as cb
from sklearn.linear_model import Lasso
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import torch

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
data_dir=os.getcwd()
train_dir=data_dir+"/train.csv"
test_dir=data_dir+"/test.csv"
test=pd.read_csv(test_dir)
train=pd.read_csv(train_dir)
#original 이 1이 아닌것 쳐내기
train = train[train['original'] == 1]
train.fillna(train.min(), inplace=True)
#test
test = test[test['original'] == 1]
test.fillna(test.min(), inplace=True)
#person 수 세기
person_count = 0
for value in train['timestamp(hr)']:
    if value == 0:
        person_count += 1

#mean, median 값 구하기
mean_value = train['timestamp(day)'].mean()
median_value = train['timestamp(day)'].median()

#corr, avearge, std
train_34 = train.drop(columns=['timestamp(day)','timestamp(hr)','original'])
describe_train = train_34.describe()
corr_train = train_34.corr()
#test
test_34 = test.drop(columns=['timestamp(day)','timestamp(hr)','original'])
describe_test = test_34.describe()
corr_test = test_34.corr()

# sns.heatmap(corr_train, annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()


#lasso 사용 alpha=0.0004일때
lasso = Lasso(alpha=0.0004)  

X = train_34.iloc[:, :-1]  
y = train_34.iloc[:, -1] 
lasso.fit(X, y)
preproc_train1=train_34.drop(train.columns[[0,1,7,8]],axis=1)
preproc_test=test_34.drop(test.columns[[0,1,7,8]],axis=1)


#2부분
y_train=train['timestamp(day)']
label_counts = {}

for label in y_train:
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] = 1

print(label_counts)
scaler = MinMaxScaler(feature_range=(0, 200))

# smote = SMOTE(sampling_strategy={0:800000,-1: 100000, -2: 100000, -3:33000 ,-4:33000,-5:33000})
# smote_train,smote_label= smote.fit_resample(preproc_train1,y_train)
# y_train=smote_label
# preproc_train= scaler.fit_transform(smote_train)

preproc_train= scaler.fit_transform(preproc_train1)

model = CatBoostClassifier()

param={'max_depth':7,#6
       'learning_rate':0.04, #0.03
       'n_estimators': 4000, # 1000
       'grow_policy': 'SymmetricTree' #'SymmetricTree', 'Lossguide', 'Depthwise'
       }
model.set_params(**param)

#model.fit(preproc_train, y_train)
true_label = test['timestamp(day)']
predictions = model.predict(preproc_test)
predictions[predictions<-3]=-3
true_label[true_label<-3]=-3
y_train[y_train<-3]=-3

#cb_model_predict= np.argmax(predictions,axis=1)
# print(cb_model_predict)
# print(cb_model_predict.shape)
# for i in range(0,predictions.shape[0]):
#     if predictions[i]!=1:
#         print(predictions[i])
# print(model.get_all_params())

print("Accuracy: %.2f"%(accuracy_score(true_label,predictions)*100),"%")

f1 = f1_score(true_label, predictions, average='macro')
print(f"F1 score: {f1}")
