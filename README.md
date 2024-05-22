# 지하철 혼잡도 예측해보기
## 목차
1. [Members](#members)
2. [Proposal](#i-proposal)
3. [Datasets](#ii-datasets)
4. [Methodology](#iii-methodology)
5. [Evaluation & Analysis](#iv-evaluation--analysis)
6. [Related Work](#v-related-work)

## Members
- 강민성 | 한양대 컴퓨터소프트웨어학부
- 김승윤 | 한양대 경영학부
- 오세원 | 한양대 실내건축디자인학과
- 황윤영 | 한양대 경영학부
  
## I. Proposal
### Motivation
서울은 교통체증이 다른 지역에 비해 심하며, 대중교통 수단이 다른 지역에 비해 잘 발달해 있다. 서울 및 수도권은 전국 중 지하철 이용 비율이 가장 높은 지역이기도 하다. 통학 및 통근자 중 지하철을 이용하는 비중은 상당히 크다. 서울시 열린데이터광장 제공 데이터에 따르면, 2023년 지하철만을 이용한 통근·통학 비율은 12.9%, 지하철+버스 이용 통근·통학 비율은 18.8%, 승용차+지하철 이용 통근·통학 비율은 1.5%이다. 특히 출퇴근 시간대에는 '지옥철'이라고 부를 정도로 사람들이 발 디딜 틈도 없을 만큼 탑승하며, 혼잡하다. 이러한 문제는 도시 생활의 질에 큰 영향을 미친다. 지하철 혼잡도를 분석하고 예측하여 어떤 시간대에 어느 역이 혼잡한 지 알 수 있다면, 그에 맞게 지하철 이용 시간이나 경로를 조정함으로써 혼잡한 지하철을 피하고 승차 편의성을 높일 수 있을 것이다.

### Goal
이 프로젝트는 2022년 서울 지하철의 다양한 데이터를 분석하고 시각화하여, 시간대별, 요일별로 승하차 인원 및 환승 인원의 패턴을 파악한다. 이러한 데이터를 학습시킨 것을 바탕으로 지하철 혼잡도를 예측하는 것을 목적으로 한다. 이를 통해 지하철 이용에 편의를 제공하며, 교통 혼잡 문제 해결을 위한, 지하철 운영 효율성을 높이기 위한 인사이트를 제공한다.

## II. Datasets
### Datasets
* 데이터셋 링크
```
서울교통공사 역별 일별 시간대별 승하차인원 정보 : http://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do
서울교통공사 환승역 환승인원정보 : http://data.seoul.go.kr/dataList/OA-12033/S/1/datasetView.do
서울교통공사 지하철혼잡도정보 : http://data.seoul.go.kr/dataList/OA-12928/F/1/datasetView.do
```

### Dataset 전처리
1. 필요한 라이브러리 가져오기 및 GPU/CPU 디바이스 설:
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)
print(torch.cuda.get_device_name(0))
```
2. 혼잡도 및 승하차 인원 데이터 로드하기
* 2022년 지하철 혼잡도와 승하차 인원 데이터를 csv파일에서 가져온다.
``` python
congestion = pd.read_csv("서울교통공사_지하철혼잡도정보_20221231.csv", encoding='cp949')
station = pd.read_csv("서울교통공사_역별 일별 시간대별 승하차인원 정보_20221231.csv", encoding='cp949')
```
* 데이터 타입을 확인한다.
``` python
print(station.dtypes)
```
3. 역명 정리하기
* 병기역명/부역명을 제거하고, 4호선 이수역과 7호선 총신대입구역은 사실상 같은 역이기 때문에, 명칭을 '총신대입구'로 통일한다.
``` python
import re
station['역명'] = station['역명'].apply(lambda x: re.sub(r'\(.*\)', '', x).strip())
station['역명'] = station['역명'].replace('이수', '총신대입구')
```
4. 환승 인원 데이터 로드
* 2022년 지하철 역별 요일별 환승인원 데이터를 csv파일에서 가져온다.
``` python
transfer = pd.read_csv("서울교통공사_역별요일별환승인원_20221231.csv", encoding='cp949')
```
5. 날짜 형식 변환 및 요일 정보 추가하기
* '수송일자' column을 날짜 형식으로 변환하고, 요일 정보를 추가한다.
``` python
station['수송일자'] = pd.to_datetime(station['수송일자'])
station['day_of_week'] = station['수송일자'].dt.dayofweek
station['day_type'] = station['day_of_week'].apply(lambda x: 'Weekday' if x < 5 else ('Saturday' if x == 5 else 'Sunday'))
```
6. Dataframe 재구성하기
* 'station' dataframe을 melt하여 시간대별 승하차 인원을 세분화하고, 그룹화하여 평균을 계산한다.
``` python
melted_df = pd.melt(station, id_vars=['호선', '역번호', '역명', '승하차구분', 'day_type'], 
                    value_vars=['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', 
                                '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', 
                                '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대',
                                '23-24시간대', '24시이후'],
                    var_name='hour', value_name='passenger_count')

grouped_df = melted_df.groupby(['호선', '역번호', '역명', '승하차구분', 'hour', 'day_type'])['passenger_count'].mean().reset_index()
pivot_df = grouped_df.pivot_table(index=['호선', '역번호', '역명', 'hour'], columns=['승하차구분', 'day_type'], values='passenger_count').reset_index()

pivot_df.columns = pivot_df.columns.map('_'.join)
pivot_df.columns = [col.rstrip('_') for col in pivot_df.columns]
pivot_df.to_csv('processed_passenger_data.csv', index=False, encoding='cp949')
```
7. 통합 데이터 로드 및 처리하기
* 가공된 승하차 인원 데이터를 로드하고, 환승 인원 데이터를 통합한다.
``` python
station1 = pd.read_csv("processed_passenger_data.csv", encoding='cp949')
transfer['역명'] = transfer['역명'].replace('총신대입구(이수)', '총신대입구')
transfer = transfer.drop(columns='연번')
transfer = transfer.rename(columns={'평일(일평균)': '환승_Weekday', '토요일': '환승_Saturday', '일요일': '환승_Sunday'})
station_transfer = pd.merge(station1, transfer, how='outer', on='역명')
station_transfer = station_transfer.fillna(0)
station_transfer = station_transfer.sort_values('역번호')
station_transfer.to_csv('join.csv', index=False, encoding='cp949')
```
8. 환승 인원 데이터 스케일링하기
* 각 역의 환승 인원을 승하차 인원 비율에 맞춰 조정한다.
``` python
new = station_transfer.copy()
for station_name in transfer['역명']:
    selected = station_transfer.loc[station_transfer['역명'] == station_name]

    total_Saturday = selected[['승차_Saturday', '하차_Saturday']].to_numpy().sum()
    total_Sunday = selected[['승차_Sunday', '하차_Sunday']].to_numpy().sum()
    total_Weekday = selected[['승차_Weekday', '하차_Weekday']].to_numpy().sum()

    for idx, row in selected.iterrows():
        scaling_Saturday = (row['승차_Saturday'] + row['하차_Saturday']) / total_Saturday
        scaling_Sunday = (row['승차_Sunday'] + row['하차_Sunday']) / total_Sunday
        scaling_Weekday = (row['승차_Weekday'] + row['하차_Weekday']) / total_Weekday
    
        new.at[idx, '환승_Saturday'] = row['환승_Saturday'] * scaling_Saturday
        new.at[idx, '환승_Sunday'] = row['환승_Sunday'] * scaling_Sunday
        new.at[idx, '환승_Weekday'] = row['환승_Weekday'] * scaling_Weekday
```
9. 처리된 데이터를 새로운 CSV파일로 저장하기
``` python
print(new.head)
new.to_csv('join2.csv', index=False, encoding='cp949')
```

## III. Methodology
## IV. Evaluation & Analysis
## V. Related Work (e.g., existing studies)
## VI. Conclusion: Discussion
