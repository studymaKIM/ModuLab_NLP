# DLC 

### 10월 8일 - NLP Overview (전희원)

- Spacy 패키지
- NER corpus ?



### 10월 10일 - From Textual Features to Input (전희원)

-  음절 단위가 각광을 받고 있다. vocabulary로 나타내었을때 총 11172개인데 자주 쓰이는 것은 몇천개 밖에 되지 않음

- 알파벳은 charactor 개수가 24개? 라서 더 유리함

- 번역 - seq2seq

- 빈도가 중요할때 - TF-IDF와 같은 기법을 사용

- 순서가 중요하지 않은 스팸 같은 NLP 문제들은 대부분 Bag of word로 잘 풀린다

- 감성분석은 꼬아서 얘기할 수도 있고 순서가 중요하다.

- 딥러닝은 순서가 중요

- Bag of Words VS One-Hot Encoding

  - BOW(Bag Of Words)

  - - 단어를 순서에 상관없이 나열
    - 빈도수가 중요할 때  
    - 스팸필터와 같은 단순한 분류기에서 높은 성능 
    - 단어의 순서를 고려하지 않는 구조적 한계 

  - One-Hot Encoding 

  - - 각 입력차원을 하나의 입력속성(예를 들어 토큰)으로 할당하는 인코딩 작업
    - BOW 대비 원핫인코딩은 차원이 하나 더 추가되는셈
    - “성별”, “나이구간” 등 의 명목형 속성을 인코딩하는 빙식으로도 활용 
    - 사전 크기 x 문장 길이 만큼의 차원을 확보해야 되기 때문에 메모리 소모가 큼 

  - 딥러닝을 다룰때는 시퀀스와 심지어 문맥까지 고려하기 때문에 대부분 BOW는 고려하지 않고, 
    원핫인코딩을 사용

- 임베딩(embedding)

- - 고밀도벡터(dense vector) 형태로 토큰을 표현
  - 사전크기만큼 표현벡터 크기를 생성할 필요가 없어진다.
  - 두 단어의 유사도를 계산할 수 있다.

- Keras에서 embedding training이 잘 안되는 이유
  - 네거티브 샘플링, 서브샘플링이 잘 되지 않아서.
  - 경험상 서브샘플링이 굉장히 중요하더라
- FastText의 장점은 모르는 단어도 학습이 가능하다는점
- GluonNLP
- *pretrain 된 임베딩은 사용하는 이유는 가지고 있는 학습데이터가 작기 때문이다.



### 10월 17일 - NGramDetectors, CNN (김형준)





### 10월 22일 - Language Model (전희원)

- generation 하는 부분에는 language model이 들어갈 수 밖에 없다.

- language model 보다는 grammar 가 더 적절한 표현이겠지만 관용적으로 language model (LM)으로 칭한다.

- -https://github.com/oxford-cs-deepnlp-2017

  https://www.edwith.org/deepnlp 



















