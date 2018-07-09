# Design Deep Neural Network Models based on Temporal Sequentiality and Integration for Document Classification

시간적 특징인 의존성을 잘 이해하는 모델 설계를 위하여 시간적 의존성을 시간적 순차성과 통합성으로 나누고 이들을 기반으로 신경망 모듈들을 선택 및 연결함으로써 문서 분류를 위한 신경망 모델 설계를 하였다. 

![](/assets/model.PNG)

문서 분류를 위한 신경망은 내부적으로 단어와 문서 표상을 단계적으로 모델링한다. 시간적 순차성을 기반으로 단어 표상을 모델링하기 위해 워드 임베딩, 컨볼루션, 순환 모듈을 사용한다. 시간적 통합성을 기반으로 문서 표상을 모델링하기 위해 최대 풀링, 주의 메카니즘을 사용한다. 

본 연구는 2018 한국컴퓨터종합학술대회(KCC2018)에 "문서 분류를 위한 시간적 순차성과 통합성 기반 심층 신경망 모델 설계"로 발표(2018년6월21일) 및 게재가 되었음. [논문], [[발표](https://1drv.ms/p/s!AllPqyV9kKUrkXwn-OdLaVH1P_od)]


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


## Summary
* 직렬로 신경망을 확장하는 것뿐만 아니라 병렬로 확장하는 것 또한 의미있음. 


## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2018.03 ~ 2018.05
