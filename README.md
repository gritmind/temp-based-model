# Design Deep Neural Network Models based on Temporal Sequentiality and Integration for Document Classification

문서를 좀 더 잘 이해할 수 있는 새로운 신경망 모델을 설계하기 위해 시간적 특징을 시간적 순차성과 통합성과 같이 두 가지로 나눠 정의하였다. 문서 분류를 위한 신경망은 내부적으로 단어, 문장, 문서 표상을 차례대로 모델링한다. 일반적으로 모두 같은 시간적 특징을 기준으로 삼지만, 본 연구에서는 좀 더 세부적으로 단어 표상을 위해서는 시간적 순차성을, 문서 표상을 위해서는 시간적 통합성을 기준으로 삼았다. 시간적 순차성 기준으로 단어 표상을 위해 근접한 단어일수록 영향력이 크도록 짧은 문맥을 집중적으로 모델링할 수 있는 순환, 컨볼루션, 워드 임베딩을 병렬로 조합하였다. 시간적 통합성 기준으로 문서 표상을 위해 긴 문맥이라도 위치에 상관없이 중요한 단어를 찾을 수 있는 최대 풀링과 주의 메카니즘을 병렬로 조합하였다. SST 데이터셋으로 기존 모델과 비슷한 기존 모델과의 비교를, AG's News 데이터셋으로 외부 모델과의 비교를 통해 제안 모델의 유효성을 확인하였다.

![](/assets/model.PNG)

본 연구는 2018 한국컴퓨터종합학술대회(KCC2018)에 "문서 분류를 위한 시간적 순차성과 통합성 기반 심층 신경망 모델 설계"로 학회 발표(2018년6월21일) 및 논문 게재를 완료하였다. [[논문](http://www.dbpia.co.kr/Journal/ArticleDetail/NODE07503243)], [[발표](https://1drv.ms/p/s!AllPqyV9kKUrkXwn-OdLaVH1P_od)]


## Prerequisites 
We use Anaconda3-5.0.1-Linux-x86_64.sh. You can create a new vitual environment with all the dependencies in the yml files: 
`~$ conda env create -f environment-tensorflow-1.yml`. You can check python libraries for this project in those .yml files.

## Dataset
* Stanford Sentiment Treebank(SST) dataset [[download](https://drive.google.com/open?id=1_trnJGAc3GWcdR69trBxGbWkKFFVZSkx)]

## Pre-trained Word Embedding Model
* GloVe [[download](https://nlp.stanford.edu/projects/glove/)]

## Usage
0. **Select Command**: for multiple programs at once, we select commands to be executed in `1_root_vocab.py` and `2_root_model.py` (arguments for core files (i.e. commands) are described in those files)

1. **Build Vocabulary** (according to (1)dataset and (2)preprocessing-type); data description is also saved.
```
~$ python 1_root_vocab.py
```

2. **Tune Model**
```
~$ python 2_root_model.py --tune
```

3. **Test Model**
```
~$ python 2_root_model.py --test
```

## Contribution
* 시간적 의존성을 시간적 순차성과 통합성으로 분해하고 이들을 단어와 문서 표상을 모델링하는 기준으로 사용
* 컨볼루션, 순환, 주의 메카니즘, 워드 임베딩을 모두 사용하는 새로운 조합의 심층 신경망
* 직렬로 신경망을 확장하는 것뿐만 아니라 주로 병렬(concatenate)로 확장한 신경망

## 주요내용
* 문서에 대한 이해도가 높은 새로운 신경망 모델 설계
* 시간적 특징을 시간적 순차성과 시간적 통합성으로 구분해서 정의
* 다양한 신경망 모듈들을 정의된 문서와 시간적 특징에 맞게 직/병렬로 조합
   - 계층구조와 같이 단어와 문서를 단계적으로 모델링함
   - 단어 표상을 위해 시간적 순차성 기준으로 워드 임베딩, 컨볼루션, 순환 구조를 사용
   - 문서 표상을 위해 시간적 통합성 기준으로 최대 풀링, 주의 메커니즘을 사용

## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2018.03 ~ 2018.05
