# Design Deep Neural Network Models based on Temporal Sequentiality and Integration for Document Classification

시간적 특징인 의존성을 잘 이해하는 모델 설계를 위하여 시간적 의존성을 시간적 순차성과 통합성으로 나누고 이들을 기반으로 신경망 모듈들을 선택 및 연결함으로써 문서 분류를 위한 신경망 모델 설계를 하였다. 

![](/assets/model.PNG)

문서 분류를 위한 신경망은 내부적으로 단어와 문서 표상을 단계적으로 모델링한다. 시간적 순차성을 기반으로 단어 표상을 모델링하기 위해 워드 임베딩, 컨볼루션, 순환 모듈을 사용한다. 시간적 통합성을 기반으로 문서 표상을 모델링하기 위해 최대 풀링, 주의 메카니즘을 사용한다. 


## Prerequisites 
We use Anaconda3-5.0.1-Linux-x86_64.sh. You can create a new vitual environment with all the dependencies in the yml files: 
`~$ conda env create -f environment-tensorflow-1.yml`. You can check python libraries for this project in those .yml files.

## Dataset
* AG's news dataset [[download](https://drive.google.com/open?id=1XbrUZk3_PFVEp7zkZVrNgnRRlXKgNWt3)]. 
* Stanford Sentiment Treebank(SST) dataset [[download](https://drive.google.com/open?id=1_trnJGAc3GWcdR69trBxGbWkKFFVZSkx)]

## Pre-trained Word Embedding Model
* GloVe [[download](https://nlp.stanford.edu/projects/glove/)]


## Usage


## Contribution
* 
* 


## Summary
*  
*


## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2018.03 ~ 2018.05
