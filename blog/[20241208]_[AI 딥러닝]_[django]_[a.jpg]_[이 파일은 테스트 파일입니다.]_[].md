# 딥러닝 팀 프로젝트

## 주제:운동데이터 분석

### 한양대학교 이정우, 한양대학교 손민우

> 데이터셋 칼럼(973명의 헬스장 회원에 대한 데이터) 

> 나이:18 ~ 59세 분포

> 성별: 여성, 남성

> 몸무게: 40kg ~ 140kg 분포

> 키:1.5m ~ 2m 분포

> 최대 BPM (1분당 심박수):160 ~ 199 분포

> 평균 BPM:50 ~ 74 분포

> 운동 시간:0.5시간 ~ 2시간 분포

> 소모 칼로리:303 ~ 1780 분포

> 운동 타입: Strength 27%, Cardio 26%, Other(460) 47% 비율로 분포함

> 지방 백분율:10% ~ 35% 사이에 분포

> 하루당 물 섭취량:1.5L ~ 3.7L

> 매주 운동 횟수:2 ~ 5

> 운동 경험 수준(레벨1:초보자, 레벨2:중급자, 레벨3:전문가):1 ~ 3

> BMI(체질량지수):12.3 ~ 49.8

### 실행 환경


프로세서: Intel(R) Core(TM) Ultra 5 125H 3.60 GHz

설치된 RAM: 16.0GB

운영 체제: Windows 11 64bit


### 사용하는 라이브러리


numpy: 계산 및 배열 처리를 위한 라이브러리

pandas:데이터 분석을 위한 라이브러리

seaborn, matplotlib:그래프 사용을 위한 시각화 라이브러리, 데이터의 통계를 시각적으로 나
타내기 위해서 사용함.

 plotly.graph_objects:세부적인 그래프 생성을 위해 사용.

 plotly.express:산점도, 막대그래프, 선 그래프 등을 쉽게 그리기 위해 사용.

 warnings:파이썬 실행 중 발생하는 경고 메시지를 처리


### 코드 분석

(라이브러리 불러오기)

 import numpy as np //numpy라이브러리를 np라는 이름으로 불러오기

import pandas as pd //pandas라이브러리를 pd라는 이름으로 불러오기

import seaborn as sns //seaborn라이브러리를 sns라는 이름으로 불러오기

import matplotlib.pyplot as plt //matplotlib.pyplot라이브러리를 plt라는 이름으로 불러옴

import plotly.graph_objects as go //plotly.graph_objects를 go라는 이름으로 불러옴

import plotly.express as px //plotly.express라이브러리를 px라는 이름으로 불러옴

from warnings import filterwarnings //warnings에서 filterwarnings명령어를 불러옴

filterwarnings('ignore') //오류 메시지를 무시하도록 설정

(데이터프레임 할당)

 df = pd.read_csv("/content/dataset/gym_members_exercise_tracking.csv") //정해진 경
로에 있는 데이터셋을 df라는 변수에 저장함   



(데이터셋 저장 경로)

 df.sample(5) //데이터프레임에서 임의의 샘플을 5개 추출함