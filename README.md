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
1. 필요한 라이브러리 가져오기 및 GPU/CPU 디바이스 설정:
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
<br>
  
2. 혼잡도 및 승하차 인원 데이터 로드하기
* 2022년 지하철 혼잡도와 승하차 인원 데이터를 csv파일에서 가져온다.
    ``` python
    station = pd.read_csv("서울교통공사_역별 일별 시간대별 승하차인원 정보_20221231.csv", encoding='cp949')
station.head
    ```
* 데이터 타입을 확인한다.
    ``` python
    print(station.dtypes)
    ```
<br>

3. 역명 정리하기
* 병기역명/부역명을 제거하고, 4호선 이수역과 7호선 총신대입구역은 사실상 같은 역이기 때문에, 명칭을 '총신대입구'로 통일한다. 그리고 서울교통공사 주관이 아니라 데이터가 없는 특정 역들을 제거한다.
  ``` python
  import re
  station['역명'] = station['역명'].apply(lambda x: re.sub(r'\(.*\)', '', x).strip())
  station['역명'] = station['역명'].replace('이수', '총신대입구')
  stations_to_remove = ['까치울', '부천시청', '부평구청', '상동', '신내']
  incheon = station[station['역명'].isin(stations_to_remove)].index
  station.drop(incheon, inplace=True)
  ```
<br>

4. 환승 인원 데이터 로드 및 날짜 처리하기
* 2022년 지하철 역별 요일별 환승인원 데이터를 csv파일에서 가져온다.
* '수송일자' column을 datetime 형식으로 변환하고, 요일을 나타내는 컬럼과 요일을 평일, 토요일, 일요일로 나눠 추가한다.
  ``` python
  transfer = pd.read_csv("서울교통공사_역별요일별환승인원_20221231.csv", encoding='cp949')
  transfer.head
  station['수송일자'] = pd.to_datetime(station['수송일자'])
  station['day_of_week'] = station['수송일자'].dt.dayofweek
  station['day_type'] = station['day_of_week'].apply(lambda x: 'Weekday' if x < 5 else   
  ('Saturday' if x == 5 else 'Sunday'))
  ```
<br>
  
5. Dataframe 재구성하기
* 시간대별 승하차 인원 데이터를 하나의 컬럼으로 melt하여 변환한다.
* 역, 호선, 시간대, 승하차 구분, 요일 유형별 평균 승하차 인원을 계산하고, 피벗 테이블로 변환하여 정리한다.
```python
  hours = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', ...]
  melted_df = pd.melt(station, id_vars=['호선', '역번호', '역명', '승하차구분', 'day_type'],
                      value_vars=hours, var_name='hour', value_name='passenger_count')
  grouped_df = melted_df.groupby(['호선', '역번호', '역명', '승하차구분', 'hour', 'day_type'])['passenger_count'].mean().reset_index()
  pivot_df = grouped_df.pivot_table(index=['호선', '역번호', '역명', 'hour'], columns=['승하차구분', 'day_type'], values='passenger_count').reset_index()
```
6. 특정 역의 승하차량 보정
*충무로역의 3호선 승하차량이 모두 4호선의 데이터로 집계되어있어, 3,4호선 각각의 승하차량 비율에 따라 나눈다.
```python
  rate_3 = 1304648 / 2420033
  rate_4 = 1115385 / 2420033
  Chungmuro = pivot_df.loc[(pivot_df['역명'] == '충무로') & (pivot_df['호선'] == 4)]
  columns = ['승차_Saturday', '승차_Sunday', '승차_Weekday', '하차_Saturday', '하차_Sunday', '하차_Weekday']

  for idx, row in Chungmuro.iterrows():
      for col in columns:
          pivot_df.at[idx - 720, col] = row[col] * rate_3
      for col in columns:
          pivot_df.at[idx, col] = row[col] * rate_4
```
*연신내역의 6호선 승하차량이 모두 3호선의 데이터로 집계되어있어, 3,6호선 각각의 승하차량 비율에 따라 나눈다.
```python
  rate_3 = 1115385 / 1708420
  rate_6 = 593035 / 1708420
  Yeonsinnae = pivot_df.loc[(pivot_df['역명'] == '연신내') & (pivot_df['호선'] == 3)]
  
  for idx, row in Yeonsinnae.iterrows():
      for col in columns:
          pivot_df.at[idx + 2360, col] = row[col] * rate_6
      for col in columns:
          pivot_df.at[idx, col] = row[col] * rate_3
```

7. 데이터 병합 및 환승 인원 보정
* 승하차 인원 데이터와 역번호 데이터를 병합한다.
* 환승 인원 데이터를 불러와 컬럼 이름을 변경하고 필요없는 데이터를 제거한다. 
```python
  station_number = pd.read_csv("station_number.csv", encoding='cp949')
  pivot_df.drop(columns='역번호', inplace=True)
  pivot_df = pd.merge(pivot_df, station_number, how='inner', on=['호선','역명'])
  station1 = pivot_df.copy()
  transfer['역명'] = transfer['역명'].replace('총신대입구(이수)', '총신대입구')
  transfer.drop(columns='연번', inplace=True)
  transfer = transfer.rename(columns={'평일(일평균)': '환승_Weekday', '토요일': '환승_Saturday', '일요일': '환승_Sunday'})
  transfer.drop(transfer[transfer['역명'] == '신내'].index, inplace=True)
```

8. 환승 인원 스케일링
* 역별 승하차 인원에 비례하여 환승 인원을 스케일링한다.
```python
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
  
  new.to_csv('join2.csv', index=False, encoding='cp949')
```

9. 혼잡도 데이터 불러오기 및 처리
```python
  #load congestion rate of year 2022
  congestion = pd.read_csv("서울교통공사_지하철혼잡도정보_20221231.csv", encoding='cp949')
  congestion.head
```
10. 상행선/하행선 구분명 정리 및 역명 통일
```python
  stations_to_remove = ['진접', '오남', '별내별가람', '신내']
  remove_index = congestion[congestion['출발역'].isin(stations_to_remove)].index
  congestion.drop(remove_index, inplace=True)
  congestion["상하구분"] = congestion["상하구분"].replace("내선", "상선")
  congestion["상하구분"] = congestion["상하구분"].replace("외선", "하선")
  congestion["출발역"] = congestion["출발역"].replace("신촌(지하)", "신촌")
  congestion["출발역"] = congestion["출발역"].replace("신천", "잠실새내")
  congestion["출발역"] = congestion["출발역"].replace("올림픽공원(한국체대)", "올림픽공원")
```

11. 시간대별 혼잡도 데이터 정리
*시간대별 혼잡도 데이터를 'hours' 배열에 맞춰 새롭게 정리하고 저장한다.
```python
  congestion1 = congestion.copy()
  time = ['5시30분', '6시00분', '6시30분', '7시00분', '7시30분', '8시00분', '8시30분', '9시00분', '9시30분', '10시00분', '10시30분', '11시00분', '11시30분', '12시00분', '12시30분', '13시00분', '13시30분', '14시00분', '14시30분', '15시00분', '15시30분', '16시00분', '16시30분', '17시00분', '17시30분', '18시00분', '18시30분', '19시00분', '19시30분', '20시00분', '20시30분', '21시00분', '21시30분', '22시00분', '22시30분', '23시00분', '23시30분', '00시00분', '00시30분']
  hours = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대','11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']
  congestion1.drop(columns=time, inplace=True)
  congestion1.drop(columns='연번', inplace=True)
  for hour in hours:
      congestion1[hour] = pd.Series(dtype='float64')
  for idx, row in congestion.iterrows():
      congestion1.at[idx, hours[0]] = row[time[0]]
      for i in range(1, 20):
          congestion1.at[idx, hours[i]] = (row[time[2*i-1]] + row[time[2*i]]) / 2
  congestion1.to_csv('congestion1.csv', index=False, encoding='cp949')
```

12. 
*
```python
  congestion2 = congestion1.melt(id_vars=['요일구분', '호선', '역번호', '출발역', '상하구분'],
                      var_name='시간대', value_name='이용객수')
  
  # Pivot the DataFrame to create new columns based on direction and day type
  congestion3 = congestion2.pivot_table(
      index=['호선', '역번호', '출발역', '시간대'],
      columns=['상하구분', '요일구분'],
      values='이용객수'
  )
  
  # Rename columns to match the specified format
  congestion3.columns = ['_'.join(col).strip() for col in congestion3.columns.values]
  congestion3 = congestion3.reset_index()
  
  congestion3 = congestion3.fillna(0)
  congestion3.rename(columns = {'출발역' : '역명', '시간대' : 'hour', '상선_공휴일' : '상선_Sunday', '상선_토요일': '상선_Saturday', '상선_평일' : '상선_Weekday', '하선_공휴일' : '하선_Sunday', '하선_토요일': '하선_Saturday', '하선_평일' : '하선_Weekday'}, inplace = True)
  congestion3.drop(columns='역번호', inplace=True)
  congestion3 = pd.merge(congestion3, station_number, how='inner', on=['호선','역명'])
  
  congestion3.to_csv('congestion3.csv', index=False, encoding='cp949')
```

13. dafk
*
```python
  final = pd.merge(new, congestion3, how='inner', on=['호선', '역명', 'hour', '역번호'])
  col = ['호선', '역번호', '역명', 'hour', '승차_Weekday', '승차_Saturday', '승차_Sunday', '하차_Weekday', '하차_Saturday', '하차_Sunday', '환승_Weekday', '환승_Saturday', '환승_Sunday', 'interval_Weekday', 'interval_Saturday', 'interval_Sunday', 'capacity', '상선_Weekday', '상선_Saturday', '상선_Sunday', '하선_Weekday', '하선_Saturday', '하선_Sunday']
  final = final[col]
  final.to_csv('2022_final.csv', index=False, encoding='cp949')
```
## III. Methodology
## IV. Evaluation & Analysis
## V. Related Work (e.g., existing studies)
*Time Series Forecasting using Pytorch
  -https://www.geeksforgeeks.org/time-series-forecasting-using-pytorch/
*Multivariate Time Series Forecasting Using LSTM
  -https://medium.com/@786sksujanislam786/multivariate-time-series-forecasting-using-lstm-4f8a9d32a509

## VI. Conclusion: Discussion
