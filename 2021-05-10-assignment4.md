{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "2017250033 이재현_머신러닝 과제.ipynb",
      "provenance": [],
      "collapsed_sections": [
        "bctqmvQQPjP-",
        "DtcqJokHWJT8"
      ]
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U3UO3NTzOAct"
      },
      "source": [
        "2017250033 이재현"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yhIT2A_xOt7n"
      },
      "source": [
        "**기본 설정**\n",
        "* 필수 모듈 불러오기\n",
        "* 그래프 출력 관련 기본 설정 지정"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XcaOnTZNO1Zp"
      },
      "source": [
        "# 파이썬 ≥3.5 필수\n",
        "import sys\n",
        "assert sys.version_info >= (3, 5)\n",
        "\n",
        "# 사이킷런 ≥0.20 필수\n",
        "import sklearn\n",
        "assert sklearn.__version__ >= \"0.20\"\n",
        "\n",
        "# 공통 모듈 임포트\n",
        "import numpy as np\n",
        "import os\n",
        "\n",
        "# 노트북 실행 결과를 동일하게 유지하기 위해\n",
        "np.random.seed(42)\n",
        "\n",
        "# 깔끔한 그래프 출력을 위해\n",
        "%matplotlib inline\n",
        "import matplotlib as mpl\n",
        "import matplotlib.pyplot as plt\n",
        "mpl.rc('axes', labelsize=14)\n",
        "mpl.rc('xtick', labelsize=12)\n",
        "mpl.rc('ytick', labelsize=12)\n",
        "\n",
        "# 그림을 저장할 위치\n",
        "PROJECT_ROOT_DIR = \".\"\n",
        "CHAPTER_ID = \"training_linear_models\"\n",
        "IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, \"images\", CHAPTER_ID)\n",
        "os.makedirs(IMAGES_PATH, exist_ok=True)\n",
        "\n",
        "def save_fig(fig_id, tight_layout=True, fig_extension=\"png\", resolution=300):\n",
        "    path = os.path.join(IMAGES_PATH, fig_id + \".\" + fig_extension)\n",
        "    print(\"그림 저장:\", fig_id)\n",
        "    if tight_layout:\n",
        "        plt.tight_layout()\n",
        "    plt.savefig(path, format=fig_extension, dpi=resolution)\n",
        "    \n",
        "# 어레이 데이터를 csv 파일로 저장하기\n",
        "def save_data(fileName, arrayName, header=''):\n",
        "    np.savetxt(fileName, arrayName, delimiter=',', header=header, comments='')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bctqmvQQPjP-"
      },
      "source": [
        "# 과제 1\n",
        "조기 종료를 사용한 배치 경사 하강법으로 로지스틱 회귀를 구현하라. 단 사이킷런을 전혀 사용하지 않아야 한다."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mSUOS31yOJ7o"
      },
      "source": [
        "##단계 1: 데이터 준비\n",
        "\n",
        "붓꽃 데이터셋의 꽃잎 길이와 꽃잎 너비(petal width) 특성만 이용한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eLSCWnyhODFe"
      },
      "source": [
        "from sklearn import datasets\n",
        "iris = datasets.load_iris()\n",
        "X = iris[\"data\"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이\n",
        "y = (iris[\"target\"] == 2).astype(np.int)  # 버지니카(Virginica) 품종일 때 1(양성)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JYC_25BeLuz-"
      },
      "source": [
        "모든 샘플에 편향을 추가한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JJZ85Qb14G_c"
      },
      "source": [
        "X_with_bias = np.c_[np.ones([len(X), 1]), X]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3_OBW7MyQeuv"
      },
      "source": [
        "결과를 일정하게 유지하기 위해 랜덤 시드를 지정합니다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "51ApMPjbQgXw"
      },
      "source": [
        "np.random.seed(2042)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eZn0FXQlQh-N"
      },
      "source": [
        "##단계 2: 테이터셋 분할\n",
        "\n",
        "데이터셋을 훈련, 검증, 테스트 용도로 6대 2대 2의 비율로 무작위로 분할한다.\n",
        "* 훈련 세트 : 60%\n",
        "* 검증 세트 : 20%\n",
        "* 테스트 세트 : 20%"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PmsG5V2qJQ63"
      },
      "source": [
        "아래 코드는 사이킷런의 train_test_split() 함수를 사용하지 않고 수동으로 무작위 분할하는 방법을 보여준다. 먼저 각 세트의 크기를 결정한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IO8AFrktQhfc"
      },
      "source": [
        "test_ratio = 0.2                                         # 테스트 세트 비율 = 20%\n",
        "validation_ratio = 0.2                                   # 검증 세트 비율 = 20%\n",
        "total_size = len(X_with_bias)                            # 전체 데이터셋 크기\n",
        "\n",
        "test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%\n",
        "validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%\n",
        "train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kZSrQV-9Q9su"
      },
      "source": [
        "np.random.permutation() 함수를 이용하여 인덱스를 무작위로 섞는다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7WfJqEu2Q10P"
      },
      "source": [
        "rnd_indices = np.random.permutation(total_size)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mVFBIJldRA9x"
      },
      "source": [
        "인덱스가 무작위로 섞였기 때문에 무작위로 분할하는 효과를 얻는다. 방법은 섞인 인덱스를 이용하여 지정된 6:2:2의 비율로 훈련, 검증, 테스트 세트로 분할하는 것이다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VAT38d1AQ9Io"
      },
      "source": [
        "X_train = X_with_bias[rnd_indices[:train_size]]\n",
        "y_train = y[rnd_indices[:train_size]]\n",
        "\n",
        "X_valid = X_with_bias[rnd_indices[train_size:-test_size]]\n",
        "y_valid = y[rnd_indices[train_size:-test_size]]\n",
        "\n",
        "X_test = X_with_bias[rnd_indices[-test_size:]]\n",
        "y_test = y[rnd_indices[-test_size:]]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ieeYTTwlRHt3"
      },
      "source": [
        "##단계 3 : 타깃 변환\n",
        "\n",
        "타깃은 0, 1로 설정되어 있다. 차례대로 버니지카 아닌 품종, 버지니카 품종을 가리킨다. 훈련 세트의 첫 5개 샘플의 품종은 다음과 같다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pRuX-AfnRGoi",
        "outputId": "84710dc5-ac1f-4d58-960e-7606a4a5a888"
      },
      "source": [
        "y_train[:5]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 1, 0, 0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LoZKWuFVkUDr"
      },
      "source": [
        "훈련세트를 90 * 1 행열로 바꾸고\n",
        "\n",
        "검증세트를 30 * 1 행열로 바꾼다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zE5nFHV94aA4"
      },
      "source": [
        "y_train = np.reshape(y_train,(90,1))\n",
        "y_valid = np.reshape(y_valid,(30,1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4Ajycq3zkfGm"
      },
      "source": [
        "##단계 4 : 로지스틱 함수 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "fLSjLzQ4MAxO"
      },
      "source": [
        "def logistic(logits):\n",
        "    return 1/(1 + np.exp(-logits))                   # 시그모이드 함수 "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lryvMEXxk24e"
      },
      "source": [
        "## 단계 5 : 경사하강법 활용 훈련"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "l-kmJy-i4cSQ"
      },
      "source": [
        "n_inputs = X_train.shape[1]           # 특성 수(n) + 1, 붓꽃의 경우: 특성 2개 + 1"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uUscaQgun0Oi"
      },
      "source": [
        "파라미터 세타를 무작위로 초기 설정한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4_HoVX0E4eaR"
      },
      "source": [
        "Theta = np.random.randn(n_inputs, 1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "a7axvBrgn5Mh"
      },
      "source": [
        "배치 경사하강법"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X1ec_l8yME8R",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f158d3b8-c78d-4c50-97f9-06ea39cc2b5b"
      },
      "source": [
        "eta = 0.01\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "\n",
        "for iteration in range(n_iterations):     # 5001번 반복 훈련\n",
        "    logits = X_train.dot(Theta)\n",
        "    Y_proba = logistic(logits)\n",
        "\n",
        "    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력\n",
        "       loss = -1/m*(np.sum(y_train * np.log(Y_proba + epsilon) + (1 - y_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "       print(iteration, loss)\n",
        "\n",
        "    error = Y_proba - y_train     # 그레이디언트 계산.\n",
        "    gradients = 1/m * X_train.T.dot(error)\n",
        "    \n",
        "    Theta = Theta - eta * gradients       # 파라미터 업데이트"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 0.881719331611068\n",
            "500 0.5881524243915477\n",
            "1000 0.5012378959005775\n",
            "1500 0.44505705967197223\n",
            "2000 0.406199640853622\n",
            "2500 0.37771576339220453\n",
            "3000 0.35583883448622555\n",
            "3500 0.3384047678040506\n",
            "4000 0.3240982139684052\n",
            "4500 0.31207879910187286\n",
            "5000 0.30178600738948846\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "772hpLc5n8MM"
      },
      "source": [
        "학습된 파라미터는 다음과 같다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ZmH0q8LlV-2K",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "05100a39-98a2-4e01-a2f6-e47f2c2656fa"
      },
      "source": [
        "Theta"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-3.54053434],\n",
              "       [ 0.10800284],\n",
              "       [ 1.87232983]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hH2tuSSIoWDM"
      },
      "source": [
        "검증 세트에 대한 예측과 정확도는 다음과 같다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "E37ou2vioGUs",
        "outputId": "ae742d6b-1d3d-47e5-d8d6-a58d1f47dd0c"
      },
      "source": [
        "logits = X_valid.dot(Theta)              \n",
        "Y_proba = logistic(logits)\n",
        "y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 확률이 0.5 이상이면 1, 아니면 0으로 표현\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9666666666666667"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5ZZ8aBtLoZp1"
      },
      "source": [
        "##단계 6 : 규제가 추가된 경사하강법 활용 훈련"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "S5sdZ1vGWABa",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "902cdb65-ab1f-41be-c0e0-b7b332ceb230"
      },
      "source": [
        "eta = 0.01\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1        # 규제 하이퍼파라미터\n",
        "\n",
        "Theta = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    logits = X_train.dot(Theta)\n",
        "    Y_proba = logistic(logits)\n",
        "    \n",
        "    if iteration % 500 == 0:\n",
        "        xentropy_loss = -1/m*(np.sum(y_train * np.log(Y_proba + epsilon) + (1 - y_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))  # 편향은 규제에서 제외\n",
        "        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실\n",
        "        print(iteration, loss)\n",
        "    \n",
        "    error = Y_proba - y_train\n",
        "    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta[1:]]   # l2 규제 그레이디언트\n",
        "    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients\n",
        "    \n",
        "    Theta = Theta - eta * gradients"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 1.786828011557888\n",
            "500 0.6482925837552796\n",
            "1000 0.5487774956653535\n",
            "1500 0.4930467279515948\n",
            "2000 0.45760168029521797\n",
            "2500 0.43301889366224006\n",
            "3000 0.41499023819981745\n",
            "3500 0.4012603391795303\n",
            "4000 0.3905134106604305\n",
            "4500 0.3819204233840173\n",
            "5000 0.37493027038065746\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "w8x5L2uNovZM"
      },
      "source": [
        "검증 세트에 대한 정확도"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VlJSj_z1WBr5",
        "outputId": "93394336-c6c6-4f10-c605-b7c7d7f15ca2"
      },
      "source": [
        "logits = X_valid.dot(Theta)              \n",
        "Y_proba = logistic(logits)\n",
        "y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_valid)\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9666666666666667"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2Art9IOGoyaD"
      },
      "source": [
        "## 단계 7 : 조기 종료 추가\n",
        "위 규제가 사용된 모델의 훈련 과정에서 매 에포크마다 검증 세트에 대한 손실을 계산하여 오차가 줄어들다가 증가하기 시작할 때 멈추도록 한다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fw82T0AVWTZK",
        "outputId": "394358d3-b49c-4ecd-de02-4c8cce611918"
      },
      "source": [
        "eta = 0.1 \n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1            # 규제 하이퍼파라미터\n",
        "best_loss = np.infty   # 최소 손실값 기억 변수\n",
        "\n",
        "Theta = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    # 훈련 및 손실 계산\n",
        "    logits = X_train.dot(Theta)\n",
        "    Y_proba = logistic(logits)\n",
        "    error = Y_proba - y_train\n",
        "    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta[1:]]\n",
        "    Theta = Theta - eta * gradients\n",
        "\n",
        "    # 검증 세트에 대한 손실 계산\n",
        "    logits = X_valid.dot(Theta)\n",
        "    Y_proba = logistic(logits)\n",
        "    xentropy_loss = -1/m*(np.sum(y_valid * np.log(Y_proba + epsilon) + (1 - y_valid ) * np.log(1 - Y_proba + epsilon)))\n",
        "    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))\n",
        "    loss = xentropy_loss + alpha * l2_loss\n",
        "    \n",
        "    # 500 에포크마다 검증 세트에 대한 손실 출력\n",
        "    if iteration % 500 == 0:\n",
        "        print(iteration, loss)\n",
        "        \n",
        "    # 에포크마다 최소 손실값 업데이트\n",
        "    if loss < best_loss:\n",
        "        best_loss = loss\n",
        "    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료\n",
        "        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력\n",
        "        print(iteration, loss, \"조기 종료!\")\n",
        "        break"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 0.3894074110194923\n",
            "500 0.1469946162902557\n",
            "685 0.14562413751155925\n",
            "686 0.1456241714788169 조기 종료!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9EcYC0svpO4C"
      },
      "source": [
        "검증 세트에 대한 정확도"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LB582o5d6654",
        "outputId": "0eddd58b-3742-4315-ddf6-064206e62ce4"
      },
      "source": [
        "logits = X_valid.dot(Theta)              \n",
        "Y_proba = logistic(logits)\n",
        "y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_valid)\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9666666666666667"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CEOmEMo6pR0b"
      },
      "source": [
        "## 단계 8 : 테스트 평가\n",
        "마지막으로 테스트 세트에 대한 모델의 최종 성능을 정확도로 측정한다. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SD5zBan5WeEI",
        "outputId": "80ff25a3-ab7b-46e9-f682-1733ea666cb2"
      },
      "source": [
        "logits = X_test.dot(Theta)              \n",
        "Y_proba = logistic(logits)\n",
        "y_predict = np.where(Y_proba >= 0.5, 1, 0)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "y_predict=np.reshape(y_predict,(-1))\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_test)\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9333333333333333"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DtcqJokHWJT8"
      },
      "source": [
        "# 과제 2\n",
        "과제 1에서 구현된 로지스틱 회귀 알고리즘에 일대다(OvR) 방식을 적용하여 붓꽃에 대한 다중 클래스 분류 알고리즘을 구현하라. 단 사이킷런을 전혀 사용하지 않아야 한다. "
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HNMXhquvqP0A"
      },
      "source": [
        "## 단계 1 : 데이터 준비\n",
        "붓꽃 데이터셋의 꽃잎 길이와 꽃잎 너비 특성만 이용한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cuLGq6ugWMF0"
      },
      "source": [
        "from sklearn import datasets\n",
        "iris = datasets.load_iris()\n",
        "X = iris[\"data\"][:, (2, 3)]  # 꽃잎 길이, 꽃잎 넓이\n",
        "y = iris[\"target\"]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sMe-nLdtqXwR"
      },
      "source": [
        "모든 샘플에 편향을 추가한다. 이유는 아래 수식을 행렬 연산으로 보다 간단하게 처리하기 위해 0번 특성값 \n",
        "x\n",
        "0\n",
        "이 항상 1이라고 가정하기 때문이다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xTJ-wtxmGNhO"
      },
      "source": [
        "X_with_bias = np.c_[np.ones([len(X), 1]), X]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_v3fOeDhqfUC"
      },
      "source": [
        "결과를 일정하게 유지하기 위해 랜덤 시드를 지정합니다. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "q0SwPAD3e8_3"
      },
      "source": [
        "np.random.seed(2042)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hTrVQbPRqmm3"
      },
      "source": [
        "## 단계 2 : 데이터셋 분할\n",
        "데이터셋을 훈련, 검증, 테스트 용도로 6대 2대 2의 비율로 무작위로 분할한다.\n",
        "* 훈련 세트: 60%\n",
        "* 검증 세트: 20%\n",
        "* 테스트 세트: 20%"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2QAHq9hCqzcp"
      },
      "source": [
        "아래 코드는 사이킷런의 train_test_split() 함수를 사용하지 않고 수동으로 무작위 분할하는 방법을 보여준다. 먼저 각 세트의 크기를 결정한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y2UH9GeXe98n"
      },
      "source": [
        "test_ratio = 0.2                                         # 테스트 세트 비율 = 20%\n",
        "validation_ratio = 0.2                                   # 검증 세트 비율 = 20%\n",
        "total_size = len(X_with_bias)                            # 전체 데이터셋 크기\n",
        "\n",
        "test_size = int(total_size * test_ratio)                 # 테스트 세트 크기: 전체의 20%\n",
        "validation_size = int(total_size * validation_ratio)     # 검증 세트 크기: 전체의 20%\n",
        "train_size = total_size - test_size - validation_size    # 훈련 세트 크기: 전체의 60%"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "P4XDLIcdq1ui"
      },
      "source": [
        "np.random.permutation() 함수를 이용하여 인덱스를 무작위로 섞는다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5ErkK-Mne_Ci"
      },
      "source": [
        "rnd_indices = np.random.permutation(total_size)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZZtkvxOLq4ZF"
      },
      "source": [
        "인덱스가 무작위로 섞였기 때문에 무작위로 분할하는 효과를 얻는다. 방법은 섞인 인덱스를 이용하여 지정된 6:2:2의 비율로 훈련, 검증, 테스트 세트로 분할하는 것이다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m8p7Y1W-fAIA"
      },
      "source": [
        "X_train = X_with_bias[rnd_indices[:train_size]]\n",
        "y_train = y[rnd_indices[:train_size]]\n",
        "\n",
        "X_valid = X_with_bias[rnd_indices[train_size:-test_size]]\n",
        "y_valid = y[rnd_indices[train_size:-test_size]]\n",
        "\n",
        "X_test = X_with_bias[rnd_indices[-test_size:]]\n",
        "y_test = y[rnd_indices[-test_size:]]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "W_zixoG9q7_R"
      },
      "source": [
        "## 단계 3 : 타깃 변환\n",
        "타깃은 0, 1, 2로 설정되어 있다. 차례대로 세토사, 버시컬러, 버지니카 품종을 가리킨다. 훈련 세트의 첫 5개 샘플의 품종은 다음과 같다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ctQ3PxejfB_g",
        "outputId": "4c5e0a21-d42f-48ef-e1e4-d1e4b98933ac"
      },
      "source": [
        "y_train[:5]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 2, 1, 1])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3OdjSWRtr_hj"
      },
      "source": [
        "학습을 위해 타깃을 원-핫 벡터로 변환해야 한다. 이유는 소프트맥스 회귀는 샘플이 주어지면 각 클래스별로 속할 확률을 구하고 구해진 결과를 실제 확률과 함께 이용하여 비용함수를 계산하기 때문이다.\n",
        "\n",
        "붓꽃 데이터의 경우 세 개의 품종 클래스별로 속할 확률을 계산해야 하기 때문에 품종을 0, 1, 2 등의 하나의 숫자로 두기 보다는 해당 클래스는 1, 나머지는 0인 확률값으로 이루어진 어레이로 다루어야 소프트맥스 회귀가 계산한 클래스별 확률과 연결된다.\n",
        "\n",
        "아래 함수 to_one_hot() 함수는 길이가 m이면서 0, 1, 2로 이루어진 1차원 어레이가 입력되면 (m, 3) 모양의 원-핫 벡터를 반환한다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GHNadRF7fC64"
      },
      "source": [
        "def to_one_hot(y):\n",
        "    n_classes = y.max() + 1                 # 클래스 수\n",
        "    m = len(y)                              # 샘플 수\n",
        "    Y_one_hot = np.zeros((m, n_classes))    # (샘플 수, 클래스 수) 0-벡터 생성\n",
        "    Y_one_hot[np.arange(m), y] = 1          # 샘플 별로 해당 클래스의 값만 1로 변경. (넘파이 인덱싱 활용)\n",
        "    return Y_one_hot"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iBfk5l-UsDTb"
      },
      "source": [
        "샘플 5개에 대해 잘 작동하는 것을 확인할 수 있다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X_B-UpnOfEAO",
        "outputId": "e93d72e1-5aa0-4dc5-b341-9a41768b1e36"
      },
      "source": [
        "y_train[:5]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 2, 1, 1])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r1o0cI1GfFEY",
        "outputId": "f2a85c15-a59d-4291-d12d-63b06f31e4a3"
      },
      "source": [
        "to_one_hot(y_train[:5])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1., 0., 0.],\n",
              "       [0., 1., 0.],\n",
              "       [0., 0., 1.],\n",
              "       [0., 1., 0.],\n",
              "       [0., 1., 0.]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 11
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r7gK-nuCsGpw"
      },
      "source": [
        "이제 훈련/검증/테스트 세트의 타깃을 모두 원-핫 벡터로 변환한다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "CqsmoTZsfGA_"
      },
      "source": [
        "Y_train_one_hot = to_one_hot(y_train)\n",
        "Y_valid_one_hot = to_one_hot(y_valid)\n",
        "Y_test_one_hot = to_one_hot(y_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OWvflno5sWtj"
      },
      "source": [
        "원-핫 벡터로 변환된 훈련/검증/테스트 세트를 각각 세토사, 버지칼라, 버지니카 경우의 훈련/검증/테스트로 나눠준다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Cl9WRn1ifHiU"
      },
      "source": [
        "Setosa_train=Y_train_one_hot[:,0]\n",
        "Versicolor_train = Y_train_one_hot[:,1]\n",
        "Virginica_train = Y_train_one_hot[:,2]\n",
        "\n",
        "Setosa_valid = Y_valid_one_hot[:,0]\n",
        "Versicolor_valid = Y_valid_one_hot[:,1]\n",
        "Virginica_valid = Y_valid_one_hot[:,2]\n",
        "\n",
        "Setosa_test = Y_test_one_hot[:,0]\n",
        "Versicolor_test = Y_test_one_hot[:,1]\n",
        "Virginica_test = Y_test_one_hot[:,2]"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "z5eZvdR3s0lu"
      },
      "source": [
        "세토사, 버지칼라, 버지니카 경우의 훈련/검증/테스트를 m*1행렬 형태로 바꿔준다"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eeUQw60OE2Vh"
      },
      "source": [
        "Setosa_train = np.reshape(Setosa_train,(90,1))\n",
        "Versicolor_train = np.reshape(Versicolor_train,(90,1))\n",
        "Virginica_train = np.reshape(Virginica_train,(90,1))\n",
        "\n",
        "Setosa_valid = np.reshape(Setosa_valid,(30,1))\n",
        "Versicolor_valid = np.reshape(Versicolor_valid,(30,1))\n",
        "Virginica_valid = np.reshape(Virginica_valid,(30,1))\n",
        "\n",
        "y_valid = np.reshape(y_valid,(30,1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KJntxhgBtJI-"
      },
      "source": [
        "##단계 4 : 로지스틱 함수 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rZ1S_RJh_GXw"
      },
      "source": [
        "def logistic(logits):\n",
        "    return 1/(1 + np.exp(-logits))                   # 시그모이드 함수 "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VsStBvtJtQT4"
      },
      "source": [
        "##단계 5 : 경사하강법 활용 훈련"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QuCIHi6v_HFT"
      },
      "source": [
        "n_inputs = X_train.shape[1]           # 특성 수(n) + 1, 붓꽃의 경우: 특성 2개 + 1"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c4-2anhhtVz8"
      },
      "source": [
        "세토사, 버지칼라, 버지니카 각각의 파라미터 세타를 무작위로 초기 설정한다. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "twHMWb8M_Qj9"
      },
      "source": [
        "Theta_Setosa = np.random.randn(n_inputs, 1)\n",
        "Theta_Versicolor = np.random.randn(n_inputs, 1)\n",
        "Theta_Virginica = np.random.randn(n_inputs, 1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_nR0Ljp7JVBO"
      },
      "source": [
        "버지니카 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a2_cJeRa_h8j",
        "outputId": "65ab4268-2104-49d6-c30c-edef666516b4"
      },
      "source": [
        "eta = 0.01\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "\n",
        "for iteration in range(n_iterations):     # 5001번 반복 훈련\n",
        "    logits = X_train.dot(Theta_Virginica)\n",
        "    Y_proba = logistic(logits)\n",
        "\n",
        "    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력\n",
        "       loss = -1/m*(np.sum(Virginica_train * np.log(Y_proba + epsilon) + (1 - Virginica_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "       print(iteration, loss)\n",
        "\n",
        "    error = Y_proba - Virginica_train     # 그레이디언트 계산.\n",
        "    gradients = 1/m * X_train.T.dot(error)\n",
        "    \n",
        "    Theta_Virginica = Theta_Virginica - eta * gradients       # 파라미터 업데이트"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 1.24300345889263\n",
            "500 0.7225341776177694\n",
            "1000 0.5840168541020665\n",
            "1500 0.4994640056646242\n",
            "2000 0.4444430012538986\n",
            "2500 0.4061493368052233\n",
            "3000 0.3779307885022022\n",
            "3500 0.3561651077443095\n",
            "4000 0.3387606059745092\n",
            "4500 0.32443981789478005\n",
            "5000 0.31238297396779313\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5k3eZwLfvegF"
      },
      "source": [
        "학습된 버지니카 파라미터는 다음과 같다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LDoTnTrnDv1C",
        "outputId": "22307068-cd8c-4906-e170-5e7be3db597b"
      },
      "source": [
        "Theta_Virginica"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-3.44731107],\n",
              "       [ 0.19151479],\n",
              "       [ 1.56891593]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rbG8Wdnkvk13"
      },
      "source": [
        "세토사 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lo824HA_ItHz",
        "outputId": "2c376d45-9ea8-41a0-aa32-0e65ddce595e"
      },
      "source": [
        "eta = 0.01\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "\n",
        "for iteration in range(n_iterations):     # 5001번 반복 훈련\n",
        "    logits = X_train.dot(Theta_Setosa)\n",
        "    Y_proba = logistic(logits)\n",
        "\n",
        "    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력\n",
        "       loss = -1/m*(np.sum(Setosa_train * np.log(Y_proba + epsilon) + (1 - Setosa_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "       print(iteration, loss)\n",
        "\n",
        "    error = Y_proba - Setosa_train     # 그레이디언트 계산.\n",
        "    gradients = 1/m * X_train.T.dot(error)\n",
        "    \n",
        "    Theta_Setosa = Theta_Setosa - eta * gradients       # 파라미터 업데이트"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 0.437577818353542\n",
            "500 0.31431995535530344\n",
            "1000 0.24739460329867496\n",
            "1500 0.2015867449720451\n",
            "2000 0.16903839269298135\n",
            "2500 0.14505018722266635\n",
            "3000 0.12678834326159452\n",
            "3500 0.11249465278438289\n",
            "4000 0.10104061337268973\n",
            "4500 0.09167722680238735\n",
            "5000 0.08389173872084243\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_aI1umYkvroD"
      },
      "source": [
        "학습된 세토사 파라미터는 다음과 같다."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NUqLAAozJOt2",
        "outputId": "b3a02070-8b8a-494c-9585-f9aeb9a176be"
      },
      "source": [
        "Theta_Setosa"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 3.51327004],\n",
              "       [-0.98251761],\n",
              "       [-1.58220754]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "E-DgmEg6vvr3"
      },
      "source": [
        "버지칼라 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Rvs_fcHIJe5_",
        "outputId": "1678c246-303f-4156-b046-e19ad76cf546"
      },
      "source": [
        "eta = 0.01\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "\n",
        "for iteration in range(n_iterations):     # 5001번 반복 훈련\n",
        "    logits = X_train.dot(Theta_Versicolor)\n",
        "    Y_proba = logistic(logits)\n",
        "\n",
        "    if iteration % 500 == 0:              # 500 에포크마다 손실(비용) 계산해서 출력\n",
        "       loss = -1/m*(np.sum(Versicolor_train * np.log(Y_proba + epsilon) + (1 - Versicolor_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "       print(iteration, loss)\n",
        "\n",
        "    error = Y_proba - Versicolor_train     # 그레이디언트 계산.\n",
        "    gradients = 1/m * X_train.T.dot(error)\n",
        "    \n",
        "    Theta_Versicolor = Theta_Versicolor - eta * gradients       # 파라미터 업데이트"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 1.2606929830898994\n",
            "500 0.6772281840254408\n",
            "1000 0.6495112807463842\n",
            "1500 0.6321202145095705\n",
            "2000 0.6208586756639016\n",
            "2500 0.61327529751536\n",
            "3000 0.6079464920366938\n",
            "3500 0.604036172333173\n",
            "4000 0.6010441944343213\n",
            "4500 0.5986650864781039\n",
            "5000 0.5967082592695107\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hrTR-1sWv5Ga"
      },
      "source": [
        "학습된 버지칼라 파라미터는 다음과 같다. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sgdRU35-Jtpw",
        "outputId": "258a519c-ebff-4e86-e9e0-e3c34c6a19de"
      },
      "source": [
        "Theta_Versicolor"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-1.5214239 ],\n",
              "       [ 0.39912725],\n",
              "       [-0.56169413]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0jtiL49jKD4h"
      },
      "source": [
        "검증 세트에 대한 예측과 정확도는 다음과 같다. logits, Y_proba를 검증 세트인 X_valid를 이용하여 계산한다. 예측 클래스는 Y_proba에서 가장 큰 값을 갖는 인덱스로 선택한다. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rzF7d3RAKFu9",
        "outputId": "91fd678d-a496-4183-931e-103dd8dc3560"
      },
      "source": [
        "logits_Setosa = X_valid.dot(Theta_Setosa)\n",
        "logits_Versicolor = X_valid.dot(Theta_Versicolor)\n",
        "logits_Virginica = X_valid.dot(Theta_Virginica)\n",
        "\n",
        "Y_proba_Setosa = logistic(logits_Setosa)\n",
        "Y_proba_Versicolor = logistic(logits_Versicolor)\n",
        "Y_proba_Virginica = logistic(logits_Virginica)\n",
        "\n",
        "Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))      # 각각의 Y_proba를 하나의 Y_proba로 합친다.\n",
        "y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "\n",
        "y_predict = np.reshape(y_predict,(30,1))\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8666666666666667"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oziJrWfdVr-Y"
      },
      "source": [
        "##단계 6 : 규제가 추가된 경사하강법 활용 훈련\n",
        "ℓ\n",
        "2\n",
        "  규제가 추가된 경사하강법 훈련을 구현한다. 코드는 기본적으로 동일하다. 다만 손실(비용)에 \n",
        "ℓ\n",
        "2\n",
        " 페널티가 추가되었고 그래디언트에도 항이 추가되었다(Theta의 첫 번째 원소는 편향이므로 규제하지 않습니다)."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vVg401qU161c"
      },
      "source": [
        "세토사 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZRurlIQ5IZL_",
        "outputId": "69f80960-0dd8-4c29-fff2-d7940650354c"
      },
      "source": [
        "eta = 0.1\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1        # 규제 하이퍼파라미터\n",
        "\n",
        "Theta_Setosa = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    logits = X_train.dot(Theta_Setosa)\n",
        "    Y_proba = logistic(logits)\n",
        "    \n",
        "    if iteration % 500 == 0:\n",
        "        xentropy_loss = -1/m*(np.sum(Setosa_train * np.log(Y_proba + epsilon) + (1 - Setosa_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "        l2_loss = 1/2 * np.sum(np.square(Theta_Setosa[1:]))  # 편향은 규제에서 제외\n",
        "        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실\n",
        "        print(iteration, loss)\n",
        "    \n",
        "    error = Y_proba - Setosa_train\n",
        "    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta_Setosa[1:]]   # l2 규제 그레이디언트\n",
        "    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients\n",
        "    \n",
        "    Theta_Setosa = Theta_Setosa - eta * gradients"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 3.1352627662530685\n",
            "500 0.19141554248807\n",
            "1000 0.18332032150568228\n",
            "1500 0.18256117011272638\n",
            "2000 0.18247547741818204\n",
            "2500 0.1824652228133692\n",
            "3000 0.1824639712095209\n",
            "3500 0.1824638174022222\n",
            "4000 0.18246379845716532\n",
            "4500 0.18246379612213304\n",
            "5000 0.18246379583441202\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5CyZUhwg1-5I"
      },
      "source": [
        "버지칼라 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "eudVVvQqXJpa",
        "outputId": "080e383e-5bc9-46af-ad59-8ff5327d1ef6"
      },
      "source": [
        "eta = 0.1\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1        # 규제 하이퍼파라미터\n",
        "\n",
        "Theta_Versicolor = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    logits = X_train.dot(Theta_Versicolor)\n",
        "    Y_proba = logistic(logits)\n",
        "    \n",
        "    if iteration % 500 == 0:\n",
        "        xentropy_loss = -1/m*(np.sum(Versicolor_train * np.log(Y_proba + epsilon) + (1 - Versicolor_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "        l2_loss = 1/2 * np.sum(np.square(Theta_Versicolor[1:]))  # 편향은 규제에서 제외\n",
        "        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실\n",
        "        print(iteration, loss)\n",
        "    \n",
        "    error = Y_proba - Versicolor_train\n",
        "    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta_Versicolor[1:]]   # l2 규제 그레이디언트\n",
        "    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients\n",
        "    \n",
        "    Theta_Versicolor = Theta_Versicolor - eta * gradients"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 1.7143076904453414\n",
            "500 0.6109654499109303\n",
            "1000 0.606642084457891\n",
            "1500 0.6065298575325324\n",
            "2000 0.6065265595915946\n",
            "2500 0.6065264605990892\n",
            "3000 0.6065264576158502\n",
            "3500 0.6065264575257266\n",
            "4000 0.6065264575229754\n",
            "4500 0.6065264575228864\n",
            "5000 0.6065264575228827\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fG_3cPpP2CwA"
      },
      "source": [
        "버지니카 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G93tNMNpXn1z",
        "outputId": "f8381e1f-6337-4ac6-e4e6-5636c30c46e4"
      },
      "source": [
        "eta = 0.1\n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1        # 규제 하이퍼파라미터\n",
        "\n",
        "Theta_Virginica = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    logits = X_train.dot(Theta_Virginica)\n",
        "    Y_proba = logistic(logits)\n",
        "    \n",
        "    if iteration % 500 == 0:\n",
        "        xentropy_loss = -1/m*(np.sum(Virginica_train * np.log(Y_proba + epsilon) + (1 - Virginica_train ) * np.log(1 - Y_proba + epsilon)))\n",
        "        l2_loss = 1/2 * np.sum(np.square(Theta_Virginica[1:]))  # 편향은 규제에서 제외\n",
        "        loss = xentropy_loss + alpha * l2_loss        # l2 규제가 추가된 손실\n",
        "        print(iteration, loss)\n",
        "    \n",
        "    error = Y_proba - Virginica_train\n",
        "    l2_loss_gradients = np.r_[np.zeros([1, 1]), alpha * Theta_Virginica[1:]]   # l2 규제 그레이디언트\n",
        "    gradients = 1/m * X_train.T.dot(error) + l2_loss_gradients\n",
        "    \n",
        "    Theta_Virginica = Theta_Virginica - eta * gradients"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 3.8854347934040043\n",
            "500 0.3756223322184947\n",
            "1000 0.34404146852932377\n",
            "1500 0.3355453014987563\n",
            "2000 0.3325221034686029\n",
            "2500 0.331316668103372\n",
            "3000 0.3308073341053733\n",
            "3500 0.3305848928284968\n",
            "4000 0.3304857748355467\n",
            "4500 0.3304410443122014\n",
            "5000 0.3304206913149579\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SXxSS6jiYNWZ",
        "outputId": "76c34c87-c36b-4d98-891b-36c56b030275"
      },
      "source": [
        "logits_Setosa = X_valid.dot(Theta_Setosa)\n",
        "logits_Versicolor = X_valid.dot(Theta_Versicolor)\n",
        "logits_Virginica = X_valid.dot(Theta_Virginica)\n",
        "\n",
        "Y_proba_Setosa = logistic(logits_Setosa)\n",
        "Y_proba_Versicolor = logistic(logits_Versicolor)\n",
        "Y_proba_Virginica = logistic(logits_Virginica)\n",
        "\n",
        "Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))\n",
        "y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "\n",
        "y_predict = np.reshape(y_predict,(30,1))\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8333333333333334"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rPP-kOCF2USD"
      },
      "source": [
        "##단계 7 : 조기 종료 추가\n",
        "위 규제가 사용된 모델의 훈련 과정에서 매 에포크마다 검증 세트에 대한 손실을 계산하여 오차가 줄어들다가 증가하기 시작할 때 멈추도록 한다"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8AxB_y4y3m8e"
      },
      "source": [
        "세토사 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1j2GQxrnIlEw",
        "outputId": "809c5138-ec42-431c-e1af-925b0f0dd1b2"
      },
      "source": [
        "eta = 0.05 \n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1            # 규제 하이퍼파라미터\n",
        "best_loss = np.infty   # 최소 손실값 기억 변수\n",
        "\n",
        "Theta_Setosa = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    # 훈련 및 손실 계산\n",
        "    logits = X_train.dot(Theta_Setosa)\n",
        "    Y_proba = logistic(logits)\n",
        "    error = Y_proba - Setosa_train\n",
        "    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta_Setosa[1:]]\n",
        "    Theta_Setosa = Theta_Setosa - eta * gradients\n",
        "\n",
        "    # 검증 세트에 대한 손실 계산\n",
        "    logits = X_valid.dot(Theta_Setosa)\n",
        "    Y_proba = logistic(logits)\n",
        "    xentropy_loss = -1/m*(np.sum(Setosa_valid * np.log(Y_proba + epsilon) + (1 - Setosa_valid ) * np.log(1 - Y_proba + epsilon)))\n",
        "    l2_loss = 1/2 * np.sum(np.square(Theta_Setosa[1:]))\n",
        "    loss = xentropy_loss + alpha * l2_loss\n",
        "    \n",
        "    # 500 에포크마다 검증 세트에 대한 손실 출력\n",
        "    if iteration % 500 == 0:\n",
        "        print(iteration, loss)\n",
        "        \n",
        "    # 에포크마다 최소 손실값 업데이트\n",
        "    if loss < best_loss:\n",
        "        best_loss = loss\n",
        "    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료\n",
        "        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력\n",
        "        print(iteration, loss, \"조기 종료!\")\n",
        "        break"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 0.40942983139282796\n",
            "440 0.10361951656552942\n",
            "441 0.10361953261633167 조기 종료!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hZVuWLxF5M3a"
      },
      "source": [
        "버지칼라 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R7jL4nHMKPcs",
        "outputId": "ff7556cf-3dd4-4090-e5d7-aaa201ee3e2a"
      },
      "source": [
        "eta = 0.05 \n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1            # 규제 하이퍼파라미터\n",
        "best_loss = np.infty   # 최소 손실값 기억 변수\n",
        "\n",
        "Theta_Versicolor = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    # 훈련 및 손실 계산\n",
        "    logits = X_train.dot(Theta_Versicolor)\n",
        "    Y_proba = logistic(logits)\n",
        "    error = Y_proba - Versicolor_train\n",
        "    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta_Versicolor[1:]]\n",
        "    Theta_Versicolor = Theta_Versicolor - eta * gradients\n",
        "\n",
        "    # 검증 세트에 대한 손실 계산\n",
        "    logits = X_valid.dot(Theta_Versicolor)\n",
        "    Y_proba = logistic(logits)\n",
        "    xentropy_loss = -1/m*(np.sum(Versicolor_valid * np.log(Y_proba + epsilon) + (1 - Versicolor_valid ) * np.log(1 - Y_proba + epsilon)))\n",
        "    l2_loss = 1/2 * np.sum(np.square(Theta_Versicolor[1:]))\n",
        "    loss = xentropy_loss + alpha * l2_loss\n",
        "    \n",
        "    # 500 에포크마다 검증 세트에 대한 손실 출력\n",
        "    if iteration % 500 == 0:\n",
        "        print(iteration, loss)\n",
        "        \n",
        "    # 에포크마다 최소 손실값 업데이트\n",
        "    if loss < best_loss:\n",
        "        best_loss = loss\n",
        "    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료\n",
        "        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력\n",
        "        print(iteration, loss, \"조기 종료!\")\n",
        "        break"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 0.416610961312532\n",
            "500 0.21643905889436313\n",
            "649 0.2160019181913218\n",
            "650 0.21600192445725527 조기 종료!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wDU5syOI5OAG"
      },
      "source": [
        "버지니카 배치 경사하강법 구현"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8-6vkOd5Z0qw",
        "outputId": "dd635e00-b4e4-4bf1-a38a-6d678e25a621"
      },
      "source": [
        "eta = 0.05 \n",
        "n_iterations = 5001\n",
        "m = len(X_train)\n",
        "epsilon = 1e-7\n",
        "alpha = 0.1            # 규제 하이퍼파라미터\n",
        "best_loss = np.infty   # 최소 손실값 기억 변수\n",
        "\n",
        "Theta_Virginica = np.random.randn(n_inputs, 1)  # 파라미터 새로 초기화\n",
        "\n",
        "for iteration in range(n_iterations):\n",
        "    # 훈련 및 손실 계산\n",
        "    logits = X_train.dot(Theta_Virginica)\n",
        "    Y_proba = logistic(logits)\n",
        "    error = Y_proba - Virginica_train\n",
        "    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, 1]), alpha * Theta_Virginica[1:]]\n",
        "    Theta_Virginica = Theta_Virginica - eta * gradients\n",
        "\n",
        "    # 검증 세트에 대한 손실 계산\n",
        "    logits = X_valid.dot(Theta_Virginica)\n",
        "    Y_proba = logistic(logits)\n",
        "    xentropy_loss = -1/m*(np.sum(Virginica_valid * np.log(Y_proba + epsilon) + (1 - Virginica_valid ) * np.log(1 - Y_proba + epsilon)))\n",
        "    l2_loss = 1/2 * np.sum(np.square(Theta_Virginica[1:]))\n",
        "    loss = xentropy_loss + alpha * l2_loss\n",
        "    \n",
        "    # 500 에포크마다 검증 세트에 대한 손실 출력\n",
        "    if iteration % 500 == 0:\n",
        "        print(iteration, loss)\n",
        "        \n",
        "    # 에포크마다 최소 손실값 업데이트\n",
        "    if loss < best_loss:\n",
        "        best_loss = loss\n",
        "    else:                                      # 에포크가 줄어들지 않으면 바로 훈련 종료\n",
        "        print(iteration - 1, best_loss)        # 종료되지 이전 에포크의 손실값 출력\n",
        "        print(iteration, loss, \"조기 종료!\")\n",
        "        break"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0 0.5704494956493964\n",
            "500 0.16184061317591292\n",
            "1000 0.14696977605361503\n",
            "1357 0.14564294270457545\n",
            "1358 0.14564294530525182 조기 종료!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e_5KzXdu5SrT"
      },
      "source": [
        "검증 세트에 대한 정확도 검사"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qNVh0HbeaguI",
        "outputId": "96c8bce5-091e-4bc7-87be-a98e2b91f3d5"
      },
      "source": [
        "logits_Setosa = X_valid.dot(Theta_Setosa)\n",
        "logits_Versicolor = X_valid.dot(Theta_Versicolor)\n",
        "logits_Virginica = X_valid.dot(Theta_Virginica)\n",
        "\n",
        "Y_proba_Setosa = logistic(logits_Setosa)\n",
        "Y_proba_Versicolor = logistic(logits_Versicolor)\n",
        "Y_proba_Virginica = logistic(logits_Virginica)\n",
        "\n",
        "Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))\n",
        "y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "\n",
        "y_predict = np.reshape(y_predict,(30,1))\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_valid)  # 정확도 계산\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yWcYl1a95WZM"
      },
      "source": [
        "##단계 8 : 테스트 세트 평가"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "66dlSzQ6gt5v",
        "outputId": "3bb45f55-b070-4f2f-ce97-23ba19b8952a"
      },
      "source": [
        "logits_Setosa = X_test.dot(Theta_Setosa)\n",
        "logits_Versicolor = X_test.dot(Theta_Versicolor)\n",
        "logits_Virginica = X_test.dot(Theta_Virginica)\n",
        "\n",
        "Y_proba_Setosa = logistic(logits_Setosa)\n",
        "Y_proba_Versicolor = logistic(logits_Versicolor)\n",
        "Y_proba_Virginica = logistic(logits_Virginica)\n",
        "\n",
        "Y_proba = np.hstack((Y_proba_Setosa,Y_proba_Versicolor,Y_proba_Virginica))\n",
        "y_predict = np.argmax(Y_proba, axis=1)          # 가장 높은 확률을 갖는 클래스 선택\n",
        "\n",
        "accuracy_score = np.mean(y_predict == y_test)  # 정확도 계산\n",
        "accuracy_score"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.8333333333333334"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    }
  ]
}