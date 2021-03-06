# TensorFlow2 API 



목표

1. Tensorflow V2의 개요와 특징을 파악한다.
2. Tensorflow V2의 3가지 주요 API 구성방식을 이해하고 활용할 수 있다. 
3. GradienTape를 활용해 보고 좀 더 로우레벨의 딥러닝 구현방식을 이해한다. 



---



## 1. Tensorflow2의 구성과 특징 

TF2의 특징을 살펴보기 전에 먼저 TF1의 특징과 문제점에 대해서 알아본다. 

### 1) Tensorflow V1의 특징과 문제점 

TF같은 딥러닝 프레임워크가 없다면 개발자는 모델을 만들기 위해 정상적인 작동을 하는 코드를 손수 만들어야 할 것이다. 그러한 과정중 가장 어려운 단계는 backpropagation을 구현하는 일일 것이다. 모델의 구조가 복잡하지 않다면 할만하다고 볼 수 있겠지만 복잡한 구조를 갖는 모델을 구현한다면 수식에 대한 gradient를 하나하나 구해야 하기 때문에 쉽지 않은 일일 뿐더러 작성도중 실수로 인한 문제가 발생할 수도 있다. 

하지만 TF와 같은 