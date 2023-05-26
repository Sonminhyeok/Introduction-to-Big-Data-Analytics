
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
data_dir=os.getcwd()
train_dir=data_dir+"/train.csv"
test_dir=data_dir+"/test.csv"
test=pd.read_csv(test_dir)
train=pd.read_csv(train_dir)
#original 이 1이 아닌것 쳐내기
train = train[train['original'] == 1]
train.fillna(train.mean(), inplace=True)

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
# sns.heatmap(corr_train, annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()


#lasso 사용 alpha=0.0004일때
lasso = Lasso(alpha=0.0004)  

X = train_34.iloc[:, :-1]  
y = train_34.iloc[:, -1] 
lasso.fit(X, y)
preproc_train=train_34.drop(train_34.columns[[0,1,6,7,8]],axis=1)
