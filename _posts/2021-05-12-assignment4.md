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



# 과제3
A. 사진을 낮과 밤으로 분류하는 로지스틱 회귀 모델을 구현하라.

B. 사진을 낮과 밤, 실내와 실외로 분류하는 다중 레이블 분류 모델을 두 개의 로지스틱 회귀 모델을 이용하여 구현하라.

C. 과제 1에서 구현한 자신의 알고리즘과 사이킷런에서 제공하는 LogisticRegression 모델의 성능을 비교하라.

단, 모델 구현에 필요한 사진을 직접 구해야 한다. 최소 100장 이상의 사진 활용해야 한다.

A에서 직접 구현한 모델과 사이킷런의 로지스틱 모델의 성능비교를 하기위해 순서를 바꿔 A->C->B 순으로 진행한다.

## A
사진을 낮과 밤으로 분류하는 로지스틱 회귀 모델 구현

직접 수집한 이미지를 구글드라이브를 통해 다운로드받기


```python
from urllib import request
url = "https://docs.google.com/uc?export=download&id=188AvTdwBIWfOxoES0hAVP5lRMasQvzkZ"
request.urlretrieve(url,"day_night.zip")
```




    ('day_night.zip', <http.client.HTTPMessage at 0x7fcdd66eff10>)



파일을 압축해제하기


```python
import os
import zipfile

local_zip = '/content/day_night.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/content')
zip_ref.close()
```

작업에 필요한 모듈 임포트


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2 
import os 
from random import shuffle 
from tqdm import tqdm 
from PIL import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
import os
# Any results you write to the current directory are saved as output.
```

label과 train, test에 따라 경로 지정하기


```python
train_day = "day_night/train/day"
train_night= "day_night/train/night"
test_day= "day_night/test/day"
test_night= "day_night/test/night"
image_size = 128
```

위의 과정이 제대로 되었는지 확인하기 위해 시험삼아 이미지를 불러온다.


```python
Image.open("day_night/train/day/day120.jpg")
```




    
![png](output_120_0.png)
    




```python
Image.open("day_night/train/night/night120.jpg")
```




    
![png](output_121_0.png)
    



수집되어있는 사진들은 사이즈가 모두 제각각이다.

머신러닝은 사진의 크기에 따라 특성수를 다르게 받아들이기때문에 이를 조정해주는 작업이 필요하다.

이를 resize라하고, 코드는 아래와 같다.


```python
for image in tqdm(os.listdir(train_night)): 
    path = os.path.join(train_night, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (image_size, image_size)).flatten()   
    np_img=np.asarray(img)
    
for image2 in tqdm(os.listdir(train_day)): 
    path = os.path.join(train_day, image2)
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 
    np_img2=np.asarray(img2)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(np_img.reshape(image_size, image_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np_img2.reshape(image_size, image_size))
plt.axis('off')
plt.title("day and night in GrayScale")
```

    100%|██████████| 100/100 [00:00<00:00, 2506.65it/s]
    100%|██████████| 100/100 [00:00<00:00, 2969.38it/s]
    




    Text(0.5, 1.0, 'day and night in GrayScale')




    
![png](output_123_2.png)
    


경로에 따라 나뉘어져 있는 낮과 밤사진들을

하나의 트레이닝 셋으로 합쳐주는 과정이 필요하다.

이 과정에 데이터 라벨링이 완료된다.


```python
def train_data():
    train_data_night = [] 
    train_data_day=[]
    for image1 in tqdm(os.listdir(train_night)): 
        path = os.path.join(train_night, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        train_data_night.append(img1) 
    for image2 in tqdm(os.listdir(train_day)): 
        path = os.path.join(train_day, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        train_data_day.append(img2) 
    
    train_data= np.concatenate((np.asarray(train_data_night),np.asarray(train_data_day)),axis=0)
    return train_data 
```

같은 작업을 테스트셋에 대해서도 해준다.


```python
def test_data():
    test_data_night = [] 
    test_data_day=[]
    for image1 in tqdm(os.listdir(test_night)): 
        path = os.path.join(test_night, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        test_data_night.append(img1) 
    for image2 in tqdm(os.listdir(test_day)): 
        path = os.path.join(test_day, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        test_data_day.append(img2) 
    
    test_data= np.concatenate((np.asarray(test_data_night),np.asarray(test_data_day)),axis=0) 
    return test_data 
```

이제 트레인셋과 테스트셋 설정 해준다.

아래의 과정에서 features와 label을 분리하여 저장한다.


```python
train_data = train_data() 
test_data = test_data()
```

    100%|██████████| 100/100 [00:00<00:00, 5042.99it/s]
    100%|██████████| 100/100 [00:00<00:00, 2775.50it/s]
    100%|██████████| 49/49 [00:00<00:00, 3660.47it/s]
    100%|██████████| 49/49 [00:00<00:00, 3879.07it/s]
    


```python
x_data=np.concatenate((train_data,test_data),axis=0)
x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
```


```python
z1 = np.zeros(100)
o1 = np.ones(100)
Y_train = np.concatenate((o1, z1), axis=0)
z = np.zeros(49)
o = np.ones(49)
Y_test = np.concatenate((o, z), axis=0)
```


```python
y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
```


```python
print("X shape: " , x_data.shape)
print("Y shape: " , y_data.shape)
```

    X shape:  (298, 128, 128)
    Y shape:  (298, 1)
    

사이킷런의 train_test_spilit 활용 train, test셋 분리하기.


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
```


```python
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
print("X train flatten",x_train_flatten.shape)
print("X test flatten",x_test_flatten.shape)
```

    X train flatten (253, 16384)
    X test flatten (45, 16384)
    


```python
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_test = y_test.T
y_train = y_train.T
day_night_y_test = y_test
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
```

    x train:  (16384, 253)
    x test:  (16384, 45)
    y train:  (1, 253)
    y test:  (1, 45)
    

데이터 전처리가 완료되었다.

다음으로 로지스틱 모델을 직접 구현해준다.


```python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))
    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
```

에포크는 1500으로, 학습률을 0.01으로 지정한뒤에 학습을 시작한다.


```python
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
```

    Cost after iteration 0: nan
    Cost after iteration 100: 0.700453
    Cost after iteration 200: 0.555647
    Cost after iteration 300: 0.462679
    Cost after iteration 400: 0.390070
    Cost after iteration 500: 0.333758
    Cost after iteration 600: 0.289998
    Cost after iteration 700: 0.255797
    Cost after iteration 800: 0.228877
    Cost after iteration 900: 0.207715
    Cost after iteration 1000: 0.192459
    Cost after iteration 1100: 0.182707
    Cost after iteration 1200: 0.174281
    Cost after iteration 1300: 0.166583
    Cost after iteration 1400: 0.159520
    


    
![png](output_141_1.png)
    


    Test Accuracy: 84.44 %
    Train Accuracy: 98.02 %
    

## C
과제 1에서 구현한 자신의 알고리즘과 사이킷런에서 제공하는 LogisticRegression 모델의 성능을 비교하라.

사이킷런 LogisticRegression에 넣기위해 데이터의 형태를 맞춰준다.


```python
x_train.shape
```




    (16384, 253)




```python
y_train.shape
```




    (1, 253)




```python
y_train2 = np.array([])
for i in y_train:
  y_train2 = np.append(y_train2, np.array([i]))
```


```python
y_test2 = np.array([])
for i in y_test:
  y_test2 = np.append(y_test2, np.array([i]))
```


```python
y_train2.shape
```




    (253,)



slover값을 'saga'로 지정, multi_class를 'multinomial'로 지정하면 로지스틱모델을 이용할수있다.


```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',
                         multi_class='multinomial').fit(x_train.T, y_train2)
```


```python
clf.score(x_test.T, y_test2)
```




    0.7333333333333333




```python
pred1 = clf.predict(x_test.T)
```

직접 구현한 로지스틱 모델의 정확도는 약 84% 사이킷런에 내장된 로지스틱 모델의 정확도는 약 68%로

직접 구현한 로지스틱 모델이 더 정확함을 알 수 있다.

## B
사진을 낮과 밤, 실내와 실외로 분류하는 다중 레이블 분류 모델을 두 개의 로지스틱 회귀 모델을 이용하여 구현하라.

두 개의 로지스틱 회귀모델중, 낮과 밤을 분류하는 모델은 위에서 이미 만들었으므로

여기에서는 먼저 실내와 실외를 분류하는 로지스틱 모델을 만들도록 한다.

과정은 낮과 밤을 분류할때와 거의 유사하므로 비슷한부분은 아주 간단하게만 집고 넘어간다.

구글 드라이브에서 실내와 실외의 데이터 다운로드


```python
from urllib import request
url = "https://docs.google.com/uc?export=download&id=1II2gidsGqhsJKP2Sfpd0YRHnA0UhKUu6"
request.urlretrieve(url,"indoor_outdoor.zip")
```




    ('indoor_outdoor.zip', <http.client.HTTPMessage at 0x7fcdc0b66750>)



압축풀기


```python
import os
import zipfile

local_zip = '/content/indoor_outdoor.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/content')
zip_ref.close()
```

경로지정


```python
train_indoor = "indoor_outdoor/train/indoor"
train_outdoor= "indoor_outdoor/train/outdoor"
test_indoor= "indoor_outdoor/test/indoor"
test_outdoor= "indoor_outdoor/test/outdoor"
image_size = 128
```

제대로 다운로드가 완료되었는지 체크


```python
Image.open("indoor_outdoor/train/indoor/indoor1.jpg")
```




    
![png](output_163_0.png)
    




```python
Image.open("indoor_outdoor/train/outdoor/outdoor1.jpg")
```




    
![png](output_164_0.png)
    



사진 리사이즈


```python
for image in tqdm(os.listdir(train_indoor)): 
    path = os.path.join(train_indoor, image)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, (image_size, image_size)).flatten()   
    np_img=np.asarray(img)
    
for image2 in tqdm(os.listdir(train_outdoor)): 
    path = os.path.join(train_outdoor, image2)
    img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    img2 = cv2.resize(img2, (image_size, image_size)).flatten() 
    np_img2=np.asarray(img2)

plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(np_img.reshape(image_size, image_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(np_img2.reshape(image_size, image_size))
plt.axis('off')
plt.title("indoor and outdoor in GrayScale")
```

    100%|██████████| 100/100 [00:00<00:00, 5768.54it/s]
    100%|██████████| 100/100 [00:00<00:00, 6733.94it/s]
    




    Text(0.5, 1.0, 'indoor and outdoor in GrayScale')




    
![png](output_166_2.png)
    


트레인/데이터셋 구성하기


```python
def train_data():
    train_data_indoor = [] 
    train_data_outdoor=[]
    for image1 in tqdm(os.listdir(train_indoor)): 
        path = os.path.join(train_indoor, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        train_data_indoor.append(img1) 
    for image2 in tqdm(os.listdir(train_outdoor)): 
        path = os.path.join(train_outdoor, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        train_data_outdoor.append(img2) 
    
    train_data= np.concatenate((np.asarray(train_data_indoor),np.asarray(train_data_outdoor)),axis=0)
    return train_data 
```


```python
def test_data():
    test_data_indoor = [] 
    test_data_outdoor=[]
    for image1 in tqdm(os.listdir(test_indoor)): 
        path = os.path.join(test_indoor, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img1 = cv2.resize(img1, (image_size, image_size))
        test_data_indoor.append(img1) 
    for image2 in tqdm(os.listdir(test_outdoor)): 
        path = os.path.join(test_outdoor, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img2 = cv2.resize(img2, (image_size, image_size))
        test_data_outdoor.append(img2) 
    
    test_data= np.concatenate((np.asarray(test_data_indoor),np.asarray(test_data_outdoor)),axis=0) 
    return test_data 
```


```python
train_data = train_data() 
test_data = test_data()
```

    100%|██████████| 100/100 [00:00<00:00, 4690.36it/s]
    100%|██████████| 100/100 [00:00<00:00, 4512.09it/s]
    100%|██████████| 47/47 [00:00<00:00, 4063.08it/s]
    100%|██████████| 47/47 [00:00<00:00, 3816.25it/s]
    


```python
x_data=np.concatenate((train_data,test_data),axis=0)
x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
```


```python
z1 = np.zeros(100)
o1 = np.ones(100)
Y_train = np.concatenate((o1, z1), axis=0)
z = np.zeros(47)
o = np.ones(47)
Y_test = np.concatenate((o, z), axis=0)
```


```python
y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
```


```python
print("X shape: " , x_data.shape)
print("Y shape: " , y_data.shape)
```

    X shape:  (294, 128, 128)
    Y shape:  (294, 1)
    

사이킷런 train_test_split을 이용해 트레이닝셋과 테스트셋 분리하기


```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
```


```python
x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
print("X train flatten",x_train_flatten.shape)
print("X test flatten",x_test_flatten.shape)
```

    X train flatten (249, 16384)
    X test flatten (45, 16384)
    


```python
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_test = y_test.T
y_train = y_train.T
out_doors_y_test = y_test
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
```

    x train:  (16384, 249)
    x test:  (16384, 45)
    y train:  (1, 249)
    y test:  (1, 45)
    


```python
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,2)))
    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100,2)))
```

학습시작


```python
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
```

    Cost after iteration 0: nan
    Cost after iteration 100: 1.909631
    Cost after iteration 200: 1.647339
    Cost after iteration 300: 1.469631
    Cost after iteration 400: 1.332520
    Cost after iteration 500: 1.213623
    Cost after iteration 600: 1.099559
    Cost after iteration 700: 0.991571
    Cost after iteration 800: 0.883707
    Cost after iteration 900: 0.364562
    Cost after iteration 1000: 1.894717
    Cost after iteration 1100: 0.807344
    Cost after iteration 1200: 0.230169
    Cost after iteration 1300: 0.147862
    Cost after iteration 1400: 0.165920
    


    
![png](output_181_1.png)
    


    Test Accuracy: 62.22 %
    Train Accuracy: 97.19 %
    

결과가 62%로 과대적합 된 것으로 생각된다.

이번엔 마찬가지로 사이킷런과 성능비교를 해본다.


```python
in_out_y_train = np.array([])
for i in y_train:
  in_out_y_train = np.append(in_out_y_train, np.array([i]))
```


```python
in_out_y_test = np.array([])
for i in y_test:
  in_out_y_test = np.append(in_out_y_test, np.array([i]))
```


```python
in_out_y_test
```




    array([1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 1.,
           0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
           1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0.])




```python
from sklearn.linear_model import LogisticRegression

lg2 = LogisticRegression(penalty='none', 
                         tol=0.1, solver='saga',C = 0.5,
                         multi_class='multinomial').fit(x_train.T, in_out_y_train)
```


```python
pred2 = lg2.predict(x_test.T)
```


```python
lg2.score(x_test.T, in_out_y_test)
```




    0.6666666666666666



성능이 직접구현한것보다 좋게나왔지만 여전히 매우 좋지 않는 수치이다.

이로써 두개의 모델이 모두 준비되었으므로, 두개의 예측값을 합쳐서 하나의 array로 만들어 다중 라벨 분류를 완성한다.

마찬가지로 낮과밤, 실내실외의 라벨이 합쳐진 테스트셋과 비교하여 정확도 성능을 측정한다.


```python
multi_label_list = []
for i in range(len(pred1)):
 multi_label_list.append([pred1[i], pred2[i]]) # 낮과밤에 대한 예측결과와 실내실외에 대한 예측결과를 샘플별로 묶어서 리스트에 저장한다.
```


```python
multi_label_pred = np.array(multi_label_list) # 저장된 리스트를 array로 바꾼다.
```


```python
multi_label_test_list = []
for i in range(len(out_doors_y_test)):
 multi_label_test_list.append([day_night_y_test[0][i], out_doors_y_test[0][i]]) # 낮과밤, 실내실외에 대한 정답을 샘플별로 묶어서 리스트에 저장한다.
```


```python
multi_label_y_test = np.array(multi_label_test_list) # 저장된 리스트를 array로 바꿔준다.
```

이제 마지막으로 정확도를 측정한다.


```python
accuracy_score = np.mean(multi_label_pred == multi_label_y_test)
accuracy_score
```




    0.6444444444444445



약 64%로 다중분류모델의 성능이 별로 좋지 않았음을 확인하였다.
