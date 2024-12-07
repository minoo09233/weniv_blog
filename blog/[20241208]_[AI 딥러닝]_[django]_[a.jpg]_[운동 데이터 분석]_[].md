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

 num_records = len(df) //데이터프레임의 크기를 num_records에 할당함

num_records //크기 출력하기

### 데이터프레임 요약

 def summary(df):
 summ = pd.DataFrame(df.dtypes, columns=['data type']) //각 열의 데이터 타입을 
가져와 새로운 데이터프레임 summ에 저장함

summ['#missing'] = df.isnull().sum().values //각 열에 결측값(null)의 개수를 계산해 
추가함

summ['Duplicate'] = df.duplicated().sum() //이터프레임에서 중복된 행의 총 개수를 
계산함

summ['#unique'] = df.nunique().values // 각 열의 고유한 값의 개수를 계산해 추가함

desc = pd.DataFrame(df.describe(include='all').transpose()) //데이터프레임의 모든 
열에 대한 통계 정보를 가져와 전치(transpose)시킨 데이터프레임 desc를 생성함

summ['min'] = desc['min'].values //통계값에 최소값 추가
summ['max'] = desc['max'].values //통계값에 최대값 추가
summ['avg'] = desc['mean'].values //통계값에 평균값 추가
summ['std dev'] = desc['std'].values //통계값에 표준편차 추가
summ['top value'] = desc['top'].values //통계값에 가장 자주 나타나는 값 추가
summ['Freq'] = desc['freq'].values //통계값에 가장 자주 나타나는 값의 빈도 추가

return summ //위의 정보가 포함된 값 반환

summary(df).style.background_gradient() //반환된 데이터프레임에 그래디언트 색상을 적용해 강조효과를 줌

df.info() //크기, 열 이름, 데이터 타입, 결측값 여부 등을 확인

df.nunique() //고유 값 (중복을 제거한 유일한 값들의 집합) 개수를 계산

### 범주형 데이터의 단변량 분석
 
 import plotly.express as px

 #범주형 데이터를 포함하는 열을 선택한 데이터프레임
cat_columns = df[['Gender', 'Workout_Type', 'Workout_Frequency (days/week)', 
'Experience_Level']]

 #단일 열에 대한 단변량 분석을 수행
def univariateAnalysis_category(cols):
    print("Distribution of", cols)
    print("_" * 60)
    colors = [
        '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#7FFFD4',  
        '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',  
        '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB', 
        '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'  
    ]
    value_counts = cat_columns[cols].value_counts()
   
    # 값 빈도수를 시각화하는 막대 그래프 생성
    fig = px.bar(
        value_counts,
        x=value_counts.index,
        y=value_counts.values,
        title=f'Distribution of {cols}',
        labels={'x': 'Categories', 'y': 'Count'},
        color_discrete_sequence=[colors]
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12), 
        title_font=dict(size=30),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )
    fig.show()

     # 비율 계산
    percentage = (value_counts / value_counts.sum()) * 100
    
    # 카테고리의 값 비율(%)을 시각화하는 파이 차트 생성
  
    fig = px.pie(
        values=percentage,
        names=value_counts.index,
        labels={'names': 'Categories', 'values': 'Percentage'},
        hole=0.5,
        color_discrete_sequence=colors
    )
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f'{cols}',
        font=dict(size=18, color='white'),
        showarrow=False
    )
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=12),
        title_font=dict(size=30),
        legend=dict(x=0.9, y=0.5),
        legend_font=dict(color='white', size=12),
        width=500,  # Adjusted width
        height=400  # Adjusted height
    )
    fig.show()
    print("       ")

 for x in cat_columns:
    univariateAnalysis_category(x)


### 연속형 데이터의 히스토그램 시각화

import plotly.express as px
 from IPython.core.display import display, HTML
 colors = [
    '#FFD700',
    '#FFB6C1',
    '#32CD32'
 ]

 #히스토그램 생성 함수
def create_histplot(df, x, title, nbins=50, color_index=0):
    print("\n")
    display(HTML(f"<h1 style='text-align:center; font-size:40px; 
font-weight:bold;'>{title} Distribution</h1>")) // HTML 제목 출력
    fig = px.histogram(df, x, nbins=nbins) //지정된 열(x)의 데이터를 기반으로 히스토그
램을 생성하고 히스토그램의 빈(bin) 개수를 설정.
    fig.update_traces(marker_color=colors[color_index]) //배열에서 색상을 선택해 그래
프 색상 적용.

    #그래프 스타일링
    fig.update_layout(
        plot_bgcolor='black', //그래프 배경을 검정색으로 설정
        paper_bgcolor='black', //그래프 외부 영역 배경을 검정색으로 설정
        font_color='white' //글꼴 색상을 흰색으로 설정
    )
    fig.show() //그래프 출력

create_histplot(df, 'Weight (kg)', 'Weight (kg)', nbins=50, color_index=0)
 create_histplot(df, 'Session_Duration (hours)', 'Session_Duration (hours)', nbins=50, 
color_index=1)
 create_histplot(df, 'Calories_Burned', 'Calories_Burned', nbins=50, color_index=2)



### (그룹별 데이터 분포 시각화)
 
 import plotly.express as px
 from IPython.display import display, HTML
 
 def groupby(data, x):
 
    result = data.groupby(x).size().rename('count').reset_index()
    return result

![성별](img/gender)


### 산점도를 생성하여 데이터를 시각화

def create_scatter_plot(data, x, y, title, xaxis_title, yaxis_title, color, width=600, 
height=400):  # Reduced dimensions
    fig = px.scatter(data, x=x, y=y, size=y, color_discrete_sequence=[color])
    fig.update_traces(marker=dict(opacity=1))
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor='black',
        width=width,
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        height=height
    )
    fig.show()

 color = [
    '#FFD700',
    '#FFA500', '#00FA9A',
    '#FFB6C1', '#FF1493',
    '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
 ]

### 반복 처리를 통한 모든 열에 대해 시각화

features = ['Age',  'Height (m)', 'Max_BPM', 'Avg_BPM', 'Session_Duration (hours)', 
            'Resting_BPM', 'Fat_Percentage', 'Water_Intake (liters)']
 
 for i, feature in enumerate(features):
    display(HTML(f"<h1 style='text-align:center; font-size:40px; 
font-weight:bold;'>{feature} Distribution</h1>"))
    grouped_data = groupby(df, feature)
    create_scatter_plot(grouped_data, feature, 'count', f'{feature} Distribution', 
feature, 'Count', color[i])
 print("\n\n\n")

 ### (상관 관계 행렬 시각화)

 numeric_df = df.select_dtypes(include=['number']) //데이터프레임에서 숫자형 (정수형 
및 실수형) 데이터만 선택
correlation_matrix = numeric_df.corr() //데이터프레임의 숫자형 열 간 상관 계수를 계산
하여 상관 행렬 생성 (1: 완전한 양의 상관 관계. -1: 완전한 음의 상관 관계. 0: 상관 관계 
없음. )
 fig = go.Figure(data=go.Heatmap(z=correlation_matrix, x=correlation_matrix.columns, 
y=correlation_matrix.columns)) //행렬 시각화
fig.update_layout(title='Correlation Heatmap') //제목 정하기
fig.show() // 출력

진한 색상: 강한 상관 관계 (양의 상관 관계는 따뜻한 색, 음의 상관 관계는 차가운 색).

중간 색상: 약한 상관 관계.

밝은 색상: 상관 관계가 거의 없는 경우