#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc

#그래프에서 한글을 사용하기 위해 설정
if platform.system() == 'Darwin':
    rc('font', family="AppleGothic")
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
    rc('font', family = font_name)

#그래프에 음수를 사용하기 위함
plt.rcParams['axes.unicode_minus'] = False


# In[15]:


import os
print(os.getcwd())


# In[27]:


#student.csv 파일을 읽어오기
#이름을 인덱스로 사용
df = pd.read_csv("./data5/data/student.csv", encoding='ms949', index_col='이름')
print(df)

df.plot(kind='bar')

#위 데이터의 경우 단순한 표준화 작업만으로는 성적을 비교하는 것이 어려울 수 있음
#최대값이나 최대값-최소값으로 나눈 데이터로는 비교하기가 어려움
#이런 경우에는 표준값이나 편차값을 구해서 비교하는 것이 좋다.


# In[28]:


#위 데이터의 경우 단순한 표준화 작업만으로는 성적을 비교하는 것이 어려울 수 있음
#최대값이나 최대값-최소값을 나눈 데이터로는 비교하기가 어려움
#이런 경우에는 표준값이나 편차값을 구해서 비교하는 것이 좋다

#평균과 표준편차 구하기
kormean, korstd = df['국어'].mean(), df['국어'].std()
engmean, engstd = df['영어'].mean(), df['영어'].std()
matmean, matstd = df['수학'].mean(), df['수학'].std()

#표준값 구하기
df['국어표준값'] = (df['국어'] - kormean)/korstd
df['영어표준값'] = (df['영어'] - engmean)/engstd
df['수학표준값'] = (df['수학'] - matmean)/matstd

#편차값 구하기
df['국어편차값'] = df['국어표준값'] * 10 + 50
df['영어편차값'] = df['영어표준값'] * 10 + 50
df['수학편차값'] = df['수학표준값'] * 10 + 50

df[['국어편차값','영어편차값', '수학편차값']].plot(kind='bar')


# In[18]:


import os
print(os.getcwd())


# # 표준화

# In[29]:


auto_mpg = pd.read_csv('./data5/data/auto-mpg.csv', header=None)
auto_mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                   'acceleration', 'model year', 'origin', 'name']

#horsepower 열의 자료형을 실수로 변경
#?를 None으로 치환하고 제거한 후 자료형 변경
auto_mpg['horsepower'].replace('?', np.nan, inplace=True)
auto_mpg.dropna(subset=['horsepower'], axis=0, inplace=True)
auto_mpg['horsepower'] = auto_mpg['horsepower'].astype('float')
print(auto_mpg.head())


# In[31]:


#horsepower 열의 표준화
auto_mpg['maxhorsepower'] = auto_mpg['horsepower'] / auto_mpg['horsepower'].max()
auto_mpg['minmaxhorsepower'] = (auto_mpg['horsepower'] - auto_mpg['horsepower'].min()) / (auto_mpg['horsepower'].max() - auto_mpg['horsepower'].min())

auto_mpg.describe()


# In[37]:


# 실제 sklearn은 scikit-learn이라고 쓴다.
from sklearn import preprocessing
#스케일링을 수행할 데이터를 가져오기
x = auto_mpg[['horsepower']].values
#print(type(x))

print("평균:", np.mean(x))
print("표준편차:", np.std(x))
print("최대값:", np.max(x))
print("최소값:", np.min(x))

scaler = preprocessing.StandardScaler()
#scaler.fit(x)
#x_scaled = scaler.transform(x)  #예전에는 fit과 transform을 나눠놨는데

x_scaled = scaler.fit_transform(x)  #요즘에는 fit과 transform을 한번에 사용가능

print("평균:", np.mean(x_scaled))
print("표준편차:", np.std(x_scaled))
print("최대값:", np.max(x_scaled))
print("최소값:", np.min(x_scaled))


# # 정규화

# In[43]:


features = np.array([[1,2], [2,3],[3,8], [4,2], [7,2]])

#정규화 객체
#l1을 norm에 적용하면 맨하튼 거리 - 합치면 1
#l2를 적용하면 유클리드 거리 - 각 값을 전체 데이터를 제곱해서 더한 값의 제곱근으로 나눈 것
normalizer = preprocessing.Normalizer(norm="l2")
l2_norm = normalizer.transform(features)
print(l2_norm)


# In[44]:


#다항과 교차항 생성
features = np.array([[1,2],[2,3], [3,8], [4,2], [7,2]])
#제곱항까지의 다항을 생성 - 열의 개수가 늘어나는데
#회귀 분석을 할 때 시간의 흐름에 따라 변화가 급격해지는경우 또는
#데이터가 부족할 때 샘플 데이터를 추가하기 위해서 사용
#제곱을 하거나 곱하기를 하게되면 데이터의 특성 자체는 크게 변화하지 않기에 사용
polynomialer = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
result = polynomialer.fit_transform(features)
print(result)


# In[50]:


features = np.array([[1,2],[2,3], [3,8], [4,2], [7,2]])

#위의 데이터에 함수 적용
result1 = preprocessing.FunctionTransformer(lambda x: x + 1).transform(features)
print(result1)

df = pd.DataFrame(features, columns=["feature1", "feature2"])
print(df.apply(lambda x : x + 1).values)

from sklearn.compose import ColumnTransformer

def add_one(x):
    return x + 1

def sub_one(x):
    return x -1
result2 = ColumnTransformer([("add_one", preprocessing.FunctionTransformer(add_one, validate=True),['feature1']),
                                             ("sub_one", preprocessing.FunctionTransformer(sub_one, validate=True),['feature2'])]).fit_transform(df)
print(result2)


# In[54]:


#auto_mpg를 horsepower를 3개의 구간으로 분할

#auto_mpg['horsepower'].describe()

#경계값 찾기
count, bin_dividers = np.histogram(auto_mpg['horsepower'], bins=3)
print(count, bin_dividers)

#각 그룹에 할당할 값의 리스트
bin_names = ['저출력', '보통출력', '고출력']

auto_mpg['hp_bin'] = pd.cut(x = auto_mpg['horsepower'],
                           bins=bin_dividers,
                           labels=bin_names,
                           include_lowest=True)
print(auto_mpg[['horsepower', 'hp_bin']].head(20))


# In[55]:


#numpy에서는 그룹의 명칭을 설정하지 않고 0,1,2 처럼 인덱스로 구분한다.
result = np.digitize(auto_mpg['horsepower'], bins=[107.33333333, 168.66666667, 230.0], right=True)
print(result)


# In[58]:


#sklearn 의 binning(구간 분할)

age = np.array([[13], [30], [67], [36], [20], [33],[27], [19]])

#2개 그룹으로 분할
binarizer = preprocessing.Binarizer(threshold=30.0)
result = binarizer.transform(age)
print(result)

#여러 개의 그룹으로 분할
#4개의 그룹으로 일련번호 형태로 일정한 비율로 분할
#strategy에 uniform을 설정하면 간격을 일정하게 분할
#encode가 ordinal이면 일련번호로 그룹이 생성
#onehot을 설정하면 onehot encoding을 한 후 희소 행렬로
#onehot-dense를 정하면 onehot encoding을 한 후 밀집 행렬로
kb = preprocessing.KBinsDiscretizer(4, encode='ordinal', strategy='quantile')
result = kb.fit_transform(age)
print(result)


# In[62]:


#군집 분석을 이용한 구간 분할
from sklearn.cluster import KMeans

sample = np.array([[13, 30], [30, 40], [67, 44], [26, 24], [22, 11], [98, 28]])
df = pd.DataFrame(sample, columns = ['feature_1', 'feature_2'])
print(df)

#3개의 군집으로 분할하는 객체 생성
cluster = KMeans(3, random_state = 42)

#sample 데이터를 이용해서 훈련
cluster.fit(sample)

#sample 데이터를 가지고 예측
df['group'] = cluster.predict(sample)
print(df)


# In[73]:


# 이상치 감지

#z-score를 이용해서 이상치를 판별해주는 함수
#데이터가 12개보다 적으면 이상치가 없다고 판단합니다.

#z_socre 보정
def outliers_z_score(ys):
    #표준편차 임계값
    threshold = 3.5
    
    #평균 대신 중앙값 사용
    mean_y = np.median(ys)
    print("평균:", mean_y)
    
    #편차가 너무 크면 중앙값 사용
    stdev_y = np.median([np.abs(y-mean_y) for y in ys])
    print("표준편차:", stdev_y)
    z_scores = [0.6745 * (y - mean_y) / stdev_y for y in ys]
    print("z_score:", z_scores)
    return  np.where(np.abs(z_scores) > threshold)

features = np.array([[10, 10, 7, 6, 3], [20000, 3, 23, 12, 11]])
print(outliers_z_score(features))


# In[74]:


# 이상치 감지

#z-score를 이용해서 이상치를 판별해주는 함수
#데이터가 12개보다 적으면 이상치가 없다고 판단합니다.

#z_socre 보정
def outliers_z_score(ys):
    #표준편차 임계값
    threshold = 3.5
    
    #평균 대신 중앙값 사용
    mean_y = np.median(ys)
    print("평균:", mean_y)
    
    #편차가 너무 크면 중앙값 사용
    stdev_y = np.median([np.abs(y-mean_y) for y in ys])
    print("표준편차:", stdev_y)
    z_scores = [0.6745 * (y - mean_y) / stdev_y for y in ys]
    print("z_score:", z_scores)
    return  np.where(np.abs(z_scores) > threshold)

features = np.array([[10, 10, 7, 6, 3], [20000000, 3, 23, 12, 11]])
print(outliers_z_score(features))


# In[76]:


#IQR을 이용하는 방법
def outliers_iqr(ys):
    #1사분위 수와 3사분위 수 구하기
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    #일반적인 데이터의 하한과 상한을 구하기
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

features = np.array([[10, 10, 7, 6, 3], [20000000, 3, 23, 12, 11]])
print(outliers_iqr(features))


# In[80]:


# 일정 비율을 데이터를 이상치로 간주하기
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

features, _ = make_blobs(n_samples = 10, n_features = 2, centers=1, random_state=42)
print(features)

#첫 번째 행의 데이터를 이상치로 수정
features[0,0] = 10000
features[0,1] = 10000

#이상치 감지 객체를 생성 - 이상치 비율을 설정
outlier_detector = EllipticEnvelope(contamination=0.1)
outlier_detector.fit(features)
#이상치로 판정되면 -1을 리턴하고 그렇지 않으면 1을 리턴
#첫 번쨰 데이터만 -1이 리턴되고 나머지는 1
outlier_detector.predict(features)


# In[87]:


#이상치 처리
houses = pd.DataFrame()
houses['Price'] = [500000, 390000, 290000,5000000]
houses['Rooms'] = [2, 3, 5, 116]
houses['Feet'] = [1500, 2000, 1300, 20000]

#Rooms 값이 20보다 크면 이상치로 간주하고 특성을 추가
houses['Outlier'] = np.where(houses['Rooms'] > 20, 1, 0)
print(houses)

#Outlier의 영향을 최소화 - 특성 변환(로그 변환)
houses['Log_Feet'] = [np.log(x) for x in houses['Feet']]
print(houses)

#Outlier의 영향을 최소화 - 특성 변환(로그 변환)
imsi = pd.DataFrame(houses['Rooms'])
scaler = preprocessing.RobustScaler()
scaler.fit(imsi)
houses['Scale_Rooms'] = scaler.transform(imsi)
print(houses)


# In[91]:


#결측치 확인
import seaborn as sns
titanic = sns.load_dataset('titanic')
#titanic.info()

#None의 개수도 출력
print(titanic['age'].value_counts(dropna=False))

print(titanic['age'].isnull().sum(axis=0))


# In[96]:


# 결측치 삭제

#각 컬럼의 None의 개수 파악
print(titanic.isnull().sum(axis=0))

#결측치의 개수가 200개 이상인 컬럼을 제거
titanic_thresh = titanic.dropna(axis=1, thresh=200)
print(titanic_thresh.columns)

#결측치의 개수가 200개 이상인 컬럼을 제거 - 200개 미만인 컬럼만 필터링
result = df[['survived', 'pclass', 'sex', 'age', 'sibsp']]

#result = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp']]

#결측치인 행만 제거 - age 행이 결측치인 행을 제거
result_age = titanic.dropna(subset=['age'], how='any', axis=0)
result_age.info()


# In[105]:


import seaborn as sns

titanic = sns.load_dataset('titanic')
#None을 포함하고 있는 값을 출력
#print(titanic['embark_town'][825:831])

#embark_town 컬럼의 특성이 계절성을 갖는다면 이 경우는 앞의 값으로 채우는 것도 
#나쁘지 않은 방법입니다.
result = titanic['embark_town'].fillna(method='ffill', inplace=True)
#print(titanic['embark_town'][825:831])

#결측치가 몇개 되지 않을 때는 대표값으로 대체
#대표값으로 사용될 수 있는 데이터는 평균, 중간값, 최빈값 등
#대표값으로 변환하는 경우 많은 양의 데이터를 변경하면 분석할 때 결과가 왜곡될 수 있음ㅡ
mode = titanic['embark_town'].value_counts()
#print(mode)

#가장 많이 출현한 데이ㅓ
#print(mode.idxmax())

titanic['embark_town'].fillna(mode.idxmax(), inplace=True)
print(titanic['embark_town'][825:831])


# In[106]:


#sklearn의 SimpleImputer 이용
#객체를 만들 때 strategy 옵션에 mean, median, most_frequent, contant를 설정
#constant를 설정하면 fill_value 옵션에 채울 값을 추가해줘야 합니다.

from sklearn.impute import SimpleImputer

features = np.array([[100], [200], [300], [400], [500], [np.nan]])

simple_imputer = SimpleImputer(strategy='median')
print(simple_imputer.fit_transform(features))


# In[107]:


get_ipython().system('pip install fancyimpute')


# In[109]:


from fancyimpute import KNN

features = np.array([[100, 200], [200, 400], [300, 600], [400, 800], [200, np.nan]])
print(KNN(k=5, verbose=0). fit_transform(features))


# In[ ]:





# In[ ]:





# In[ ]:




