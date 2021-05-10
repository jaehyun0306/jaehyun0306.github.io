2017250033 이재현

**기본 설정**
* 필수 모듈 불러오기
* 그래프 출력 관련 기본 설정 지정


```python
# 파이썬 ≥3.5 필수
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
import sklearn
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트
import numpy as np
import os

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
# 어레이 데이터를 csv 파일로 저장하기
def save_data(fileName, arrayName, header=''):
    np.savetxt(fileName, arrayName, delimiter=',', header=header, comments='')
```

# 과제 1
조기 종료를 사용한 배치 경사 하강법으로 로지스틱 회귀를 구현하라. 단 사이킷런을 전혀 사용하지 않아야 한다.

##단계 1: 데이터 준비

붓꽃 데이터셋의 꽃잎 길이와 꽃잎 너비(petal width) 특성만 이용한다.


```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = (iris["target"] == 2).astype(np.int)  # 버지니카(Virginica) 품종일 때 1(양성)
```

모든 샘플에 편향을 추가한다.


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X]
```

결과를 일정하게 유지하기 위해 랜덤 시드를 지정합니다


```python
np.random.seed(2042)
```

##단계 2: 테이터셋 분할

데이터셋을 훈련, 검증, 테스트 용도로 6대 2대 2의 비율로 무작위로 분할한다.
* 훈련 세트 : 60%
* 검증 세트 : 20%
* 테스트 세트 : 20%

아래 코드는 사이킷런의 train_test_split() 함수를 사용하지 않고 수동으로 무작위 분할하는 방법을 보여준다. 먼저 각 세트의 크기를 결정한다.


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```

np.random.permutation() 함수를 이용하여 인덱스를 무작위로 섞는다.


```python
rnd_indices = np.random.permutation(total_size)
```

인덱스가 무작위로 섞였기 때문에 무작위로 분할하는 효과를 얻는다. 방법은 섞인 인덱스를 이용하여 지정된 6:2:2의 비율로 훈련, 검증, 테스트 세트로 분할하는 것이다.


```python
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```

##단계 3 : 타깃 변환

타깃은 0, 1로 설정되어 있다. 차례대로 버니지카 아닌 품종, 버지니카 품종을 가리킨다. 훈련 세트의 첫 5개 샘플의 품종은 다음과 같다.


```python
y_train[:5]
```




    array([0, 0, 1, 0, 0])



훈련세트를 90 * 1 행열로 바꾸고

검증세트를 30 * 1 행열로 바꾼다.


```python
y_train = np.reshape(y_train,(90,1))
y_valid = np.reshape(y_valid,(30,1))
```

##단계 4 : 로지스틱 함수 구현


```python
def logistic(logits):
    return 1/(1 + np.exp(-logits))                   # 시그모이드 함수 
```

## 단계 5 : 경사하강법 활용 훈련


```python
n_inputs = X_train.shape[1]           # 특성 수(n) + 1, 붓꽃의 경우: 특성 2개 + 1
```

파라미터 세타를 무작위로 초기 설정한다.


```python
Theta = np.random.randn(n_inputs, 1)
```

배치 경사하강법


```python
eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     # 5001번 반복 훈련
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)

    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력
       loss = -1/m*(np.sum(y_train * np.log(Y_proba + epsilon) + (1 - y_train ) * np.log(1 - Y_proba + epsilon)))
       print(iteration, loss)

    error = Y_proba - y_train     # 그레이디언트 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta = Theta - eta * gradients       # 파라미터 업데이트
```

    0 0.881719331611068
    500 0.5881524243915477
    1000 0.5012378959005775
    1500 0.44505705967197223
    2000 0.406199640853622
    2500 0.37771576339220453
    3000 0.35583883448622555
    3500 0.3384047678040506
    4000 0.3240982139684052
    4500 0.31207879910187286
    5000 0.30178600738948846
    

학습된 파라미터는 다음과 같다


```python
Theta
```




    array([[-3.54053434],
           [ 0.10800284],
           [ 1.87232983]])



검증 세트에 대한 예측과 정확도는 다음과 같다


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 확률이 0.5 이상이면 1, 아니면 0으로 표현

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.9666666666666667



##단계 6 : 규제가 추가된 경사하강법 활용 훈련


```python
eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1        # 규제 하이퍼파라미터

Theta = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -1/m*(np.sum(y_train * np.log(Y_proba + epsilon) + (1 - y_train ) * np.log(1 - Y_proba + epsilon)))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - y_train
    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta = Theta - eta * gradients
```

    0 1.786828011557888
    500 0.6482925837552796
    1000 0.5487774956653535
    1500 0.4930467279515948
    2000 0.45760168029521797
    2500 0.43301889366224006
    3000 0.41499023819981745
    3500 0.4012603391795303
    4000 0.3905134106604305
    4500 0.3819204233840173
    5000 0.37493027038065746
    

검증 세트에 대한 정확도


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 가장 높은 확률을 갖는 클래스 선택

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
```




    0.9666666666666667



## 단계 7 : 조기 종료 추가
위 규제가 사용된 모델의 훈련 과정에서 매 에포크마다 검증 세트에 대한 손실을 계산하여 오차가 줄어들다가 증가하기 시작할 때 멈추도록 한다


```python
eta = 0.1 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta)
    Y_proba = logistic(logits)
    error = Y_proba - y_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta)
    Y_proba = logistic(logits)
    xentropy_loss = -1/m*(np.sum(y_valid * np.log(Y_proba + epsilon) + (1 - y_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.3894074110194923
    500 0.1469946162902557
    685 0.14562413751155925
    686 0.1456241714788169 조기 종료!
    

검증 세트에 대한 정확도


```python
logits = X_valid.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 가장 높은 확률을 갖는 클래스 선택

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
```




    0.9666666666666667



## 단계 8 : 테스트 평가
마지막으로 테스트 세트에 대한 모델의 최종 성능을 정확도로 측정한다. 


```python
logits = X_test.dot(Theta)              
Y_proba = logistic(logits)
y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 가장 높은 확률을 갖는 클래스 선택
y_predict=np.reshape(y_predict,(-1))

accuracy_score = np.mean(y_predict == y_test)
accuracy_score
```




    0.9333333333333333



# 과제 2
과제 1에서 구현된 로지스틱 회귀 알고리즘에 일대다(OvR) 방식을 적용하여 붓꽃에 대한 다중 클래스 분류 알고리즘을 구현하라. 단 사이킷런을 전혀 사용하지 않아야 한다. 

## 단계 1 : 데이터 준비
붓꽃 데이터셋의 꽃잎 길이와 꽃잎 너비 특성만 이용한다.


```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이
y = iris["target"]
```

모든 샘플에 편향을 추가한다. 이유는 아래 수식을 행렬 연산으로 보다 간단하게 처리하기 위해 0번 특성값 
x
0
이 항상 1이라고 가정하기 때문이다


```python
X_with_bias = np.c_[np.ones([len(X), 1]), X]
```

결과를 일정하게 유지하기 위해 랜덤 시드를 지정합니다. 


```python
np.random.seed(2042)
```

## 단계 2 : 데이터셋 분할
데이터셋을 훈련, 검증, 테스트 용도로 6대 2대 2의 비율로 무작위로 분할한다.
* 훈련 세트: 60%
* 검증 세트: 20%
* 테스트 세트: 20%

아래 코드는 사이킷런의 train_test_split() 함수를 사용하지 않고 수동으로 무작위 분할하는 방법을 보여준다. 먼저 각 세트의 크기를 결정한다.


```python
test_ratio = 0.2                                         # 테스트 세트 비율 = 20%
validation_ratio = 0.2                                   # 검증 세트 비율 = 20%
total_size = len(X_with_bias)                            # 전체 데이터셋 크기

test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%
validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%
train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%
```

np.random.permutation() 함수를 이용하여 인덱스를 무작위로 섞는다.


```python
rnd_indices = np.random.permutation(total_size)
```

인덱스가 무작위로 섞였기 때문에 무작위로 분할하는 효과를 얻는다. 방법은 섞인 인덱스를 이용하여 지정된 6:2:2의 비율로 훈련, 검증, 테스트 세트로 분할하는 것이다.


```python
X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```

## 단계 3 : 타깃 변환
타깃은 0, 1, 2로 설정되어 있다. 차례대로 세토사, 버시컬러, 버지니카 품종을 가리킨다. 훈련 세트의 첫 5개 샘플의 품종은 다음과 같다


```python
y_train[:5]
```




    array([0, 1, 2, 1, 1])



학습을 위해 타깃을 원-핫 벡터로 변환해야 한다. 이유는 소프트맥스 회귀는 샘플이 주어지면 각 클래스별로 속할 확률을 구하고 구해진 결과를 실제 확률과 함께 이용하여 비용함수를 계산하기 때문이다.

붓꽃 데이터의 경우 세 개의 품종 클래스별로 속할 확률을 계산해야 하기 때문에 품종을 0, 1, 2 등의 하나의 숫자로 두기 보다는 해당 클래스는 1, 나머지는 0인 확률값으로 이루어진 어레이로 다루어야 소프트맥스 회귀가 계산한 클래스별 확률과 연결된다.

아래 함수 to_one_hot() 함수는 길이가 m이면서 0, 1, 2로 이루어진 1차원 어레이가 입력되면 (m, 3) 모양의 원-핫 벡터를 반환한다.


```python
def to_one_hot(y):
    n_classes = y.max() + 1                 # 클래스 수
    m = len(y)                              # 샘플 수
    Y_one_hot = np.zeros((m, n_classes))    # (샘플 수, 클래스 수) 0-벡터 생성
    Y_one_hot[np.arange(m), y] = 1          # 샘플 별로 해당 클래스의 값만 1로 변경. (넘파이 인덱싱 활용)
    return Y_one_hot
```

샘플 5개에 대해 잘 작동하는 것을 확인할 수 있다.


```python
y_train[:5]
```




    array([0, 1, 2, 1, 1])




```python
to_one_hot(y_train[:5])
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 1., 0.]])



이제 훈련/검증/테스트 세트의 타깃을 모두 원-핫 벡터로 변환한다


```python
Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)
```

원-핫 벡터로 변환된 훈련/검증/테스트 세트를 각각 세토사, 버지칼라, 버지니카 경우의 훈련/검증/테스트로 나눠준다


```python
Setosa_train=Y_train_one_hot[:,0]
Versicolor_train = Y_train_one_hot[:,1]
Virginica_train = Y_train_one_hot[:,2]

Setosa_valid = Y_valid_one_hot[:,0]
Versicolor_valid = Y_valid_one_hot[:,1]
Virginica_valid = Y_valid_one_hot[:,2]

Setosa_test = Y_test_one_hot[:,0]
Versicolor_test = Y_test_one_hot[:,1]
Virginica_test = Y_test_one_hot[:,2]
```

세토사, 버지칼라, 버지니카 경우의 훈련/검증/테스트를 m*1행렬 형태로 바꿔준다


```python
Setosa_train = np.reshape(Setosa_train,(90,1))
Versicolor_train = np.reshape(Versicolor_train,(90,1))
Virginica_train = np.reshape(Virginica_train,(90,1))

Setosa_valid = np.reshape(Setosa_valid,(30,1))
Versicolor_valid = np.reshape(Versicolor_valid,(30,1))
Virginica_valid = np.reshape(Virginica_valid,(30,1))

y_valid = np.reshape(y_valid,(30,1))
```

##단계 4 : 로지스틱 함수 구현


```python
def logistic(logits):
    return 1/(1 + np.exp(-logits))                   # 시그모이드 함수 
```

##단계 5 : 경사하강법 활용 훈련


```python
n_inputs = X_train.shape[1]           # 특성 수(n) + 1, 붓꽃의 경우: 특성 2개 + 1
```

세토사, 버지칼라, 버지니카 각각의 파라미터 세타를 무작위로 초기 설정한다. 


```python
Theta_Setosa = np.random.randn(n_inputs, 1)
Theta_Versicolor = np.random.randn(n_inputs, 1)
Theta_Virginica = np.random.randn(n_inputs, 1)
```

버지니카 배치 경사하강법 구현


```python
eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     # 5001번 반복 훈련
    logits = X_train.dot(Theta_Virginica)
    Y_proba = logistic(logits)

    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력
       loss = -1/m*(np.sum(Virginica_train * np.log(Y_proba + epsilon) + (1 - Virginica_train ) * np.log(1 - Y_proba + epsilon)))
       print(iteration, loss)

    error = Y_proba - Virginica_train     # 그레이디언트 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta_Virginica = Theta_Virginica - eta * gradients       # 파라미터 업데이트
```

    0 1.24300345889263
    500 0.7225341776177694
    1000 0.5840168541020665
    1500 0.4994640056646242
    2000 0.4444430012538986
    2500 0.4061493368052233
    3000 0.3779307885022022
    3500 0.3561651077443095
    4000 0.3387606059745092
    4500 0.32443981789478005
    5000 0.31238297396779313
    

학습된 버지니카 파라미터는 다음과 같다.


```python
Theta_Virginica
```




    array([[-3.44731107],
           [ 0.19151479],
           [ 1.56891593]])



세토사 배치 경사하강법 구현


```python
eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     # 5001번 반복 훈련
    logits = X_train.dot(Theta_Setosa)
    Y_proba = logistic(logits)

    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력
       loss = -1/m*(np.sum(Setosa_train * np.log(Y_proba + epsilon) + (1 - Setosa_train ) * np.log(1 - Y_proba + epsilon)))
       print(iteration, loss)

    error = Y_proba - Setosa_train     # 그레이디언트 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta_Setosa = Theta_Setosa - eta * gradients       # 파라미터 업데이트
```

    0 0.437577818353542
    500 0.31431995535530344
    1000 0.24739460329867496
    1500 0.2015867449720451
    2000 0.16903839269298135
    2500 0.14505018722266635
    3000 0.12678834326159452
    3500 0.11249465278438289
    4000 0.10104061337268973
    4500 0.09167722680238735
    5000 0.08389173872084243
    

학습된 세토사 파라미터는 다음과 같다.


```python
Theta_Setosa
```




    array([[ 3.51327004],
           [-0.98251761],
           [-1.58220754]])



버지칼라 배치 경사하강법 구현


```python
eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

for iteration in range(n_iterations):     # 5001번 반복 훈련
    logits = X_train.dot(Theta_Versicolor)
    Y_proba = logistic(logits)

    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력
       loss = -1/m*(np.sum(Versicolor_train * np.log(Y_proba + epsilon) + (1 - Versicolor_train ) * np.log(1 - Y_proba + epsilon)))
       print(iteration, loss)

    error = Y_proba - Versicolor_train     # 그레이디언트 계산.
    gradients = 1/m * X_train.T.dot(error)
    
    Theta_Versicolor = Theta_Versicolor - eta * gradients       # 파라미터 업데이트
```

    0 1.2606929830898994
    500 0.6772281840254408
    1000 0.6495112807463842
    1500 0.6321202145095705
    2000 0.6208586756639016
    2500 0.61327529751536
    3000 0.6079464920366938
    3500 0.604036172333173
    4000 0.6010441944343213
    4500 0.5986650864781039
    5000 0.5967082592695107
    

학습된 버지칼라 파라미터는 다음과 같다. 


```python
Theta_Versicolor
```




    array([[-1.5214239 ],
           [ 0.39912725],
           [-0.56169413]])



검증 세트에 대한 예측과 정확도는 다음과 같다. logits, Y_proba를 검증 세트인 X_valid를 이용하여 계산한다. 예측 클래스는 Y_proba에서 가장 큰 값을 갖는 인덱스로 선택한다. 


```python
logits_Setosa = X_valid.dot(Theta_Setosa)
logits_Versicolor = X_valid.dot(Theta_Versicolor)
logits_Virginica = X_valid.dot(Theta_Virginica)

Y_proba_Setosa = logistic(logits_Setosa)
Y_proba_Versicolor = logistic(logits_Versicolor)
Y_proba_Virginica = logistic(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))      # 각각의 Y_proba를 하나의 Y_proba로 합친다.
y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택

y_predict = np.reshape(y_predict,(30,1))

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.8666666666666667



##단계 6 : 규제가 추가된 경사하강법 활용 훈련
ℓ
2
  규제가 추가된 경사하강법 훈련을 구현한다. 코드는 기본적으로 동일하다. 다만 손실(비용)에 
ℓ
2
 페널티가 추가되었고 그래디언트에도 항이 추가되었다(Theta의 첫 번째 원소는 편향이므로 규제하지 않습니다).

세토사 배치 경사하강법 구현


```python
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1        # 규제 하이퍼파라미터

Theta_Setosa = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    logits = X_train.dot(Theta_Setosa)
    Y_proba = logistic(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -1/m*(np.sum(Setosa_train * np.log(Y_proba + epsilon) + (1 - Setosa_train ) * np.log(1 - Y_proba + epsilon)))
        l2_loss = 1/2 * np.sum(np.square(Theta_Setosa[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - Setosa_train
    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta_Setosa[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta_Setosa = Theta_Setosa - eta * gradients
```

    0 3.1352627662530685
    500 0.19141554248807
    1000 0.18332032150568228
    1500 0.18256117011272638
    2000 0.18247547741818204
    2500 0.1824652228133692
    3000 0.1824639712095209
    3500 0.1824638174022222
    4000 0.18246379845716532
    4500 0.18246379612213304
    5000 0.18246379583441202
    

버지칼라 배치 경사하강법 구현


```python
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1        # 규제 하이퍼파라미터

Theta_Versicolor = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    logits = X_train.dot(Theta_Versicolor)
    Y_proba = logistic(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -1/m*(np.sum(Versicolor_train * np.log(Y_proba + epsilon) + (1 - Versicolor_train ) * np.log(1 - Y_proba + epsilon)))
        l2_loss = 1/2 * np.sum(np.square(Theta_Versicolor[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - Versicolor_train
    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta_Versicolor[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta_Versicolor = Theta_Versicolor - eta * gradients
```

    0 1.7143076904453414
    500 0.6109654499109303
    1000 0.606642084457891
    1500 0.6065298575325324
    2000 0.6065265595915946
    2500 0.6065264605990892
    3000 0.6065264576158502
    3500 0.6065264575257266
    4000 0.6065264575229754
    4500 0.6065264575228864
    5000 0.6065264575228827
    

버지니카 배치 경사하강법 구현


```python
eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1        # 규제 하이퍼파라미터

Theta_Virginica = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    logits = X_train.dot(Theta_Virginica)
    Y_proba = logistic(logits)
    
    if iteration % 500 == 0:
        xentropy_loss = -1/m*(np.sum(Virginica_train * np.log(Y_proba + epsilon) + (1 - Virginica_train ) * np.log(1 - Y_proba + epsilon)))
        l2_loss = 1/2 * np.sum(np.square(Theta_Virginica[1:]))  # 편향은 규제에서 제외
        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실
        print(iteration, loss)
    
    error = Y_proba - Virginica_train
    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta_Virginica[1:]]   # l2 규제 그레이디언트
    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients
    
    Theta_Virginica = Theta_Virginica - eta * gradients
```

    0 3.8854347934040043
    500 0.3756223322184947
    1000 0.34404146852932377
    1500 0.3355453014987563
    2000 0.3325221034686029
    2500 0.331316668103372
    3000 0.3308073341053733
    3500 0.3305848928284968
    4000 0.3304857748355467
    4500 0.3304410443122014
    5000 0.3304206913149579
    


```python
logits_Setosa = X_valid.dot(Theta_Setosa)
logits_Versicolor = X_valid.dot(Theta_Versicolor)
logits_Virginica = X_valid.dot(Theta_Virginica)

Y_proba_Setosa = logistic(logits_Setosa)
Y_proba_Versicolor = logistic(logits_Versicolor)
Y_proba_Virginica = logistic(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))
y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택

y_predict = np.reshape(y_predict,(30,1))

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.8333333333333334



##단계 7 : 조기 종료 추가
위 규제가 사용된 모델의 훈련 과정에서 매 에포크마다 검증 세트에 대한 손실을 계산하여 오차가 줄어들다가 증가하기 시작할 때 멈추도록 한다

세토사 배치 경사하강법 구현


```python
eta = 0.05 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta_Setosa = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta_Setosa)
    Y_proba = logistic(logits)
    error = Y_proba - Setosa_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta_Setosa[1:]]
    Theta_Setosa = Theta_Setosa - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta_Setosa)
    Y_proba = logistic(logits)
    xentropy_loss = -1/m*(np.sum(Setosa_valid * np.log(Y_proba + epsilon) + (1 - Setosa_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta_Setosa[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.40942983139282796
    440 0.10361951656552942
    441 0.10361953261633167 조기 종료!
    

버지칼라 배치 경사하강법 구현


```python
eta = 0.05 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta_Versicolor = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta_Versicolor)
    Y_proba = logistic(logits)
    error = Y_proba - Versicolor_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta_Versicolor[1:]]
    Theta_Versicolor = Theta_Versicolor - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta_Versicolor)
    Y_proba = logistic(logits)
    xentropy_loss = -1/m*(np.sum(Versicolor_valid * np.log(Y_proba + epsilon) + (1 - Versicolor_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta_Versicolor[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.416610961312532
    500 0.21643905889436313
    649 0.2160019181913218
    650 0.21600192445725527 조기 종료!
    

버지니카 배치 경사하강법 구현


```python
eta = 0.05 
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1            # 규제 하이퍼파라미터
best_loss = np.infty   # 최소 손실값 기억 변수

Theta_Virginica = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화

for iteration in range(n_iterations):
    # 훈련 및 손실 계산
    logits = X_train.dot(Theta_Virginica)
    Y_proba = logistic(logits)
    error = Y_proba - Virginica_train
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta_Virginica[1:]]
    Theta_Virginica = Theta_Virginica - eta * gradients

    # 검증 세트에 대한 손실 계산
    logits = X_valid.dot(Theta_Virginica)
    Y_proba = logistic(logits)
    xentropy_loss = -1/m*(np.sum(Virginica_valid * np.log(Y_proba + epsilon) + (1 - Virginica_valid ) * np.log(1 - Y_proba + epsilon)))
    l2_loss = 1/2 * np.sum(np.square(Theta_Virginica[1:]))
    loss = xentropy_loss + alpha * l2_loss
    
    # 500 에포크마다 검증 세트에 대한 손실 출력
    if iteration % 500 == 0:
        print(iteration, loss)
        
    # 에포크마다 최소 손실값 업데이트
    if loss < best_loss:
        best_loss = loss
    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료
        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력
        print(iteration, loss, "조기 종료!")
        break
```

    0 0.5704494956493964
    500 0.16184061317591292
    1000 0.14696977605361503
    1357 0.14564294270457545
    1358 0.14564294530525182 조기 종료!
    

검증 세트에 대한 정확도 검사


```python
logits_Setosa = X_valid.dot(Theta_Setosa)
logits_Versicolor = X_valid.dot(Theta_Versicolor)
logits_Virginica = X_valid.dot(Theta_Virginica)

Y_proba_Setosa = logistic(logits_Setosa)
Y_proba_Versicolor = logistic(logits_Versicolor)
Y_proba_Virginica = logistic(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))
y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택

y_predict = np.reshape(y_predict,(30,1))

accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산
accuracy_score
```




    0.8



##단계 8 : 테스트 세트 평가


```python
logits_Setosa = X_test.dot(Theta_Setosa)
logits_Versicolor = X_test.dot(Theta_Versicolor)
logits_Virginica = X_test.dot(Theta_Virginica)

Y_proba_Setosa = logistic(logits_Setosa)
Y_proba_Versicolor = logistic(logits_Versicolor)
Y_proba_Virginica = logistic(logits_Virginica)

Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))
y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택

accuracy_score = np.mean(y_predict == y_test)  # 정확도 계산
accuracy_score
```




    0.8333333333333334


