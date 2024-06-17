# 지하철 혼잡도 예측해보기
## 목차
1. [Members](#members)
2. [Proposal](#i-proposal)
3. [Datasets](#ii-datasets)
4. [Methodology](#iii-methodology)
5. [Evaluation & Analysis](#iv-evaluation--analysis)
6. [Related Work](#v-related-work-eg-existing-studies)
7. [Conclusion](#vi-conclusion-discussion)
8. [Credits](#vii-credits)

<br>

## Members
- 강민성 | 한양대 컴퓨터소프트웨어학부
- 김승윤 | 한양대 경영학부
- 오세원 | 한양대 실내건축디자인학과
- 황윤영 | 한양대 경영학부

<br>

## I. Proposal
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/a3d4eca5-5541-4cef-9804-78710188b26c)
<br>

### Motivation
서울은 교통체증이 다른 지역에 비해 심하며, 대중교통 수단이 다른 지역에 비해 잘 발달해 있다. 서울 및 수도권은 전국 중 지하철 이용 비율이 가장 높은 지역이기도 하다. 통학 및 통근자 중 지하철을 이용하는 비중은 상당히 크다. 서울시 열린데이터광장 제공 데이터에 따르면, 2023년 지하철만을 이용한 통근·통학 비율은 12.9%, 지하철+버스 이용 통근·통학 비율은 18.8%, 승용차+지하철 이용 통근·통학 비율은 1.5%이다. 특히 출퇴근 시간대에는 '지옥철'이라고 부를 정도로 사람들이 발 디딜 틈도 없을 만큼 탑승하며, 혼잡하다. 이러한 문제는 도시 생활의 질에 큰 영향을 미친다. 지하철 혼잡도를 분석하고 예측하여 어떤 시간대에 어느 노선과 어느 구간이이 혼잡한 지 미리 알 수 있다면, 승객들은 이에 맞게 지하철 이용 시간이나 경로를 조정함으로써 더 쾌적하게 지하철을 이용할 수 있을 것이다.

### Goal
이 프로젝트는 2022년 서울 지하철의 다양한 데이터를 분석하고 시각화하여, 시간대별, 요일별로 승하차 인원 및 환승 인원의 패턴을 파악한다. 이러한 데이터를 학습시킨 것을 바탕으로 지하철 혼잡도를 예측하는 것을 목적으로 한다. 이를 통해 지하철 이용에 편의를 제공하며, 교통 혼잡 문제 해결을 위한, 지하철 운영 효율성을 높이기 위한 인사이트를 제공한다.

### 지하철 혼잡도란
혼잡도란 열차에 얼마나 많은 사람이 탑승했는지를 알려주는 수치로, 실제 승차 인원을 승차 정원으로 나눈 값을 말한다. 수도권 전철 1-9호선, 경의중앙선, 수인분당선 등 수도권 주요 노선에서 전동차 한 칸의 정원은 약 160명이다.
실제로는 지하철에 어떻게 적용되고 있을까?
서울교통공사에 따르면, 전동차의 각 차량마다 무게를 감지하는 하중 감지 센서가 내장돼, 이는 실시간으로 객차 내 탑승 무게를 감지하고 측정한다. 칸별 정원 160명의 무게를 기준으로 계산해 79% 이하면 ‘여유’, 80%-129% 면 ‘보통’, 130% 이상일 경우 ‘혼잡’으로 분류한다. 즉, 사람의 몸무게를 65kg으로 가정하면, 127명(약 8.2ton)보다 적은 수가 타면 ‘여유’, 128-207명(8.2-13.4ton)이 타면 ‘보통’, 208명(13.4ton)보다 많이 타면 ‘혼잡’으로 표시되는 것이다. 출·퇴근 등 매우 혼잡한 시간대엔 정원의 200%까지도 탑승한다. 한 편성당(10량) 전체 차량 중량은 343ton으로 칸별 중량은 약 34.3ton이며, 정원의 2배인 320명(약 20.8ton)이 타더라도 기계적으로 안전에 이상은 없다.
<br> (출처 : https://www.epnc.co.kr/news/articleView.html?idxno=82928) <br>
혼잡도가 우려되는 수준으로 판단해 열차 증차 등 혼잡도 완화 조치가 필요하다고 판단되는 시점은 150%이다.

<br>

## II. Datasets
### Datasets
* 데이터셋 링크
```
서울교통공사 역별 일별 시간대별 승하차인원 정보 : http://data.seoul.go.kr/dataList/OA-12921/F/1/datasetView.do
서울교통공사 환승역 환승인원정보 : http://data.seoul.go.kr/dataList/OA-12033/S/1/datasetView.do
서울교통공사 지하철혼잡도정보 : http://data.seoul.go.kr/dataList/OA-12928/F/1/datasetView.do
```

### Dataset 전처리
1. 필요한 라이브러리 가져오기
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
<br>
  
2. 혼잡도 및 승하차 인원 데이터 로드하기
* 2022년 지하철 혼잡도와 승하차 인원 데이터를 csv파일에서 가져온다.
``` python
#load number of embarking/disembarking people of each station of year 2022
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
* '수송일자' column을 datetime 형식으로 변환하고, 'day of week' column으로 새로 만든 뒤 요일을 평일, 토요일, 일요일로 분류한다.
* melt 함수를 이용하여 데이터프레임을 행당 하나의 역, 시간, 요일 유형으로 변환한다.
``` python
transfer = pd.read_csv("서울교통공사_역별요일별환승인원_20221231.csv", encoding='cp949')
transfer.head
  
station['수송일자'] = pd.to_datetime(station['수송일자'])
station['day_of_week'] = station['수송일자'].dt.dayofweek
station['day_type'] = station['day_of_week'].apply(lambda x: 'Weekday' if x < 5 else   
('Saturday' if x == 5 else 'Sunday'))

hours = ['06시이전', '06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대','11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대', '24시이후']

# Melt the dataframe to have one row per station, hour, and day type
melted_df = pd.melt(station, id_vars=['호선', '역번호', '역명', '승하차구분', 'day_type'], 
                    value_vars = hours,
                    var_name='hour', value_name='passenger_count')
```
<br>
  
5. 이상치 제거 및 역, 시간대별로 승/하차 및 요일 유형에 따른 평균 승객 수 계산하여 정리하기
* 주어진 데이터프레임에서 passenger_count 열의 이상치를 제거
* 역, 승하차 구분, 시간대, 요일 유형별로 그룹화하여 passenger_count의 평균을 계산한다.
* 피벗 테이블을 사용하여 각 역과 시간대별로 승차/하차 및 요일 유형에 따른 평균 승객 수를 정리한다.
* 시간대(hour) 열을 지정된 순서로 카테고리화하여 정렬한다.
* 역번호와 시간대별로 정렬하여 최종 데이터를 준비한다.
```python
def remove_outliers(df):
    Q1 = df['passenger_count'].quantile(0.25)
    Q3 = df['passenger_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df['passenger_count'] >= lower_bound) & (df['passenger_count'] <= upper_bound)]

# Group by station, embark/disembark, hour, and day type to calculate the mean
grouped = melted_df.groupby(['호선', '역번호', '역명', '승하차구분', 'hour', 'day_type'], group_keys=False).apply(remove_outliers)
grouped_df = grouped.groupby(['호선', '역번호', '역명', '승하차구분', 'hour', 'day_type'])['passenger_count'].mean().reset_index()

# Pivot the dataframe to get the desired format
pivot_df = grouped_df.pivot_table(index=['호선', '역번호', '역명', 'hour'], 
                                  columns=['승하차구분', 'day_type'], values='passenger_count').reset_index()

# Convert 'hour' to a categorical type with the specified order
pivot_df['hour'] = pd.Categorical(pivot_df['hour'], categories=hours, ordered=True)

# Sort by '역번호' and the categorical 'hour' column
pivot_df = pivot_df.sort_values(by=['역번호', 'hour']).reset_index(drop=True)

#pivot_df.to_csv('cleaned_station_data.csv', index=False, encoding='cp949')
```

* 피벗 테이블의 다중 인덱스 열 이름을 단일 문자열로 결합하고, 불필요한 '_'를 제거해 깔끔하게 정리한다.
```python
pivot_df.columns = pivot_df.columns.map('_'.join)
pivot_df.columns = [col.rstrip('_') for col in pivot_df.columns]
```
<br>

6. 특정 역의 승하차량 보정
* 충무로역의 3호선 승하차량이 모두 4호선의 데이터로 집계되어있어, 3,4호선 각각의 승하차량 비율에 따라 나눈다.
```python
rate_3 = 1304648 / 2420033
rate_4 = 1115385 / 2420033
Chungmuro = pivot_df.loc[(pivot_df['역명'] == '충무로') & (pivot_df['호선'] == 4)]
columns = ['승차_Saturday', '승차_Sunday', '승차_Weekday', '하차_Saturday', '하차_Sunday', '하차_Weekday']

# Iterate over the filtered rows
for idx, row in Chungmuro.iterrows():
  for col in columns:
    pivot_df.at[idx - 720, col] = row[col] * rate_3
  for col in columns:
    pivot_df.at[idx, col] = row[col] * rate_4
```
* 이와 마찬가지의 방법으로
  - 연신내역의 6호선 승하차량이 모두 3호선의 데이터로 집계되어있어, 3,6호선 각각의 승하차량 비율에 따라 나눈다.
  - 창동역의 1호선(경원선) 승하차량이 모두 4호선의 데이터로 집계되어있어, 1,4호선 각각의 승하차량 비율에 따라 나눈다.
<br>

7. 데이터 병합 및 일부 환승역 승하차량 보정
* 'station_number.csv' 파일을 읽어와 'pivot_df'와 'station_number' 데이터를 병합한다.
* 'pivot_df'의 내용을 'station1'으로 복사한다
* 'transfer' 데이터프레임에서 총신대입구역(이수) 역명을 총신대입구역으로 수정하고, 열 이름을 요일 별로 나눠 변경하고, 신내역의 데이터를 제거한다.
```python
station_number = pd.read_csv("station_number.csv", encoding='cp949')

#역번호 맞추기
pivot_df.drop(columns='역번호', inplace=True)
pivot_df = pd.merge(pivot_df, station_number, how='inner', on=['호선','역명'])

#station1 = pd.read_csv("processed_passenger_data.csv", encoding='cp949')
station1 = pivot_df.copy()
station1.head

transfer['역명'] = transfer['역명'].replace('총신대입구(이수)', '총신대입구')
transfer.drop(columns='연번', inplace=True)
transfer = transfer.rename(columns={'평일(일평균)': '환승_Weekday', '토요일': '환승_Saturday', '일요일': '환승_Sunday'})
transfer.drop(transfer[transfer['역명'] == '신내'].index, inplace=True)

transfer.head
```
<br>

* 'station1'과 'transfer' 데이터프레임을 '역명'을 기준으로 결합하여 'station_transfer'데이터프레임을 생성한다.
* 생성한 'station_transfer'와 배차간격을 나타내는 'interval'데이터프레임을 '호선'과 'hour' 컬럼을 기준으로 결합하여 'station_transfer' 데이터프레임을 수정한다.
* 'hour' 컬럼값을 카테고리형으로 변환 후 정렬된 순서형 데이터로 만든다.
* 'station_transfer'를 '역번호'와 'hour'컬럼을 기준으로 정렬하고 인덱스를 재설정하고 다시 'hour' 컬럼 값을 문자열로 변환
* 'station_transfer'의 결측값 0으로 채우기

```python
interval = pd.read_csv("interval.csv", encoding='cp949')

station_transfer = pd.merge(station1, transfer, how='outer', on='역명')
station_transfer = pd.merge(station_transfer, interval, how='outer', on=['호선', 'hour'])

station_transfer['hour'] = pd.Categorical(station_transfer['hour'], categories=hours, ordered=True)
station_transfer = station_transfer.sort_values(by=['역번호', 'hour']).reset_index(drop=True)

station_transfer['hour'] = station_transfer['hour'].astype(str)

station_transfer = station_transfer.fillna(0)

#station_transfer.to_csv('join.csv', index=False, encoding='cp949')

station_transfer.head
station_transfer.columns
station_transfer.dtypes
```

8. 환승 인원 스케일링
* 역별 승하차 인원 데이터를 이용해 환승 인원 데이터를 비율에 맞춰 스케일링하고, 최종 결과를 csv 파일로 저장한다.
```python
new = station_transfer.copy()
  
for station_name in transfer['역명']:
  selected = station_transfer.loc[station_transfer['역명'] == station_name]
  lines = pd.unique(selected['호선'])

  for line in lines:
    selected_line = selected.loc[selected['호선'] == line]

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
<br>

9. 2022년도 혼잡도 데이터 불러오기 및 처리
```python
#load congestion rate of year 2022
congestion = pd.read_csv("서울교통공사_지하철혼잡도정보_20221231.csv", encoding='cp949')

congestion.head
```
<br>

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
<br>

11. 시간대별 혼잡도 데이터 정리
* 시간대별 혼잡도 데이터를 'hours' 배열에 맞춰 새롭게 정리하고 저장한다.
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
<br>

12. 혼잡도 데이터 재구성하여 저장하기
* 'congestion1' 데이터프레임에서 요일, 호선, 역번호, 출발역, 상하구분을 기준으로 '시간대'와 '이용객수' 열을 재구성하여 새로운 데이터프레임 'congestion2'를 생성한다.
*이 데이터를 피벗하여 상하구분과 요일구분을 기준으로 새로운 열을 생성한 후, 열 이름을 지정된 형식에 맞게 변경하고 인덱스를 재설정하여 'congestion3' 데이터프레임을 준비한다.
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
```

* 'congestion3' 데이터프레임의 결측치를 0으로 채우고, 열 이름을 알아보기 쉽게 변경한다. '역번호'열은 삭제하고, 'conjestion3'데이터프레임과 'station_number'데이터프레임을 '호선'과 '역명'을 기준으로 inner join하여 합친다.
```python
congestion3 = congestion3.fillna(0)
congestion3.rename(columns = {'출발역' : '역명', '시간대' : 'hour', '상선_공휴일' : '상선_Sunday', '상선_토요일': '상선_Saturday', '상선_평일' : '상선_Weekday', '하선_공휴일' : '하선_Sunday', '하선_토요일': '하선_Saturday', '하선_평일' : '하선_Weekday'}, inplace = True)
congestion3.drop(columns='역번호', inplace=True)
congestion3 = pd.merge(congestion3, station_number, how='inner', on=['호선','역명'])

congestion3.to_csv('congestion3.csv', index=False, encoding='cp949')
```
<br>

13. 최종 데이터셋 준비하기
* 데이터프레임을 '호선', '역명', 'hour', '역번호' 열을 기준으로 조인한 후, 필요한 열을 선택하여 최종 데이터셋을 만들고, 이를 '2022_final.csv'라는 파일로 저장한다.
```python
  final = pd.merge(new, congestion3, how='inner', on=['호선', '역명', 'hour', '역번호'])
  col = ['호선', '역번호', '역명', 'hour', '승차_Weekday', '승차_Saturday', '승차_Sunday', '하차_Weekday', '하차_Saturday', '하차_Sunday', '환승_Weekday', '환승_Saturday', '환승_Sunday', 'interval_Weekday', 'interval_Saturday', 'interval_Sunday', 'capacity', '상선_Weekday', '상선_Saturday', '상선_Sunday', '하선_Weekday', '하선_Saturday', '하선_Sunday']
  final = final[col]
  final.to_csv('2022_final.csv', index=False, encoding='cp949')
```
<br>

### 데이터 시각화
1. 각 요일의(평일, 토요일, 일요일) 시간대별 승차 인원 및 상/하선 혼잡도
* 평일 시간대별 승차 및 하차인원과 상선 및 하선 혼잡도 비교 (예 : 1호선, 청량리역)

  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/24e5e20e-9a50-4b65-b80c-0a923fa67fd5) <br>
  
  - 출근 시간대인 7-8시간대에 승/하차 인원과 하선 혼잡도가 증가하는 것을 확인할 수 있다.
  - 퇴근 시간대인 18-19시간대에 승/하차 인원과 상선 혼잡도가 증가하는 것을 확인할 수 있다.


* 주말 시간대별 승차 인원 및 상/하선 혼잡도(예: 1호선, 청량리역)

  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/f55ddd07-57e8-4532-9c5e-fe83e222ede3) <br>
  
  - 주말에는 오후 시간대에 승/하차 인원이 증가함에 따라 11시간대에 하선 혼잡도가 증가하며, 16시간대에 상선 혼잡도가 증가하는 것을 확인 할 수 있다.

* 역별 승하차 인원, 역별 상/하선 혼잡도 (예: 1호선, 평일, 07-08 시간대)

  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/fcd2fe7a-cb53-44b8-8efd-d1d0f6ab382d) <br>

  - 평일 혼잡도가 높은 시간대인 7-8시간대에 역별 승/하차 인원과 혼잡도를 비교하였다. 그래프를 통해 청량리역에서 승차한 뒤, 목적지인 서울역에서 내리는 사람이 많을 것으로 추정하였다. <br> 
  따라서 동묘앞역이나, 동대문역에서는 승하차 인원은 적지만 혼잡도가 높게 유지되는 것을 알 수 있고, 최종 도착지인 서울역에서 승/하차 인원이 높은 것에 비해 혼잡도가 감소하는 것을 확인할 수 있다. <br> 
  즉, 승/하차 인원과 혼잡도는 직접적인 관계를 찾을 수 없지만, 이전 역에서 승/하차 인원이 많다면 내부 혼잡도가 증가한 채로 유지한다고 생각할 수 있다. <br>
  따라서, 선형적인 회귀 방법보다는 LSTM을 이용하여 시간대별 데이터를 전체적으로 입력받아 혼잡도를 유추해 보았다.


2. 각 요일의(평일, 토요일, 일요일) 시간대별 배차간격 및 혼잡도
* 평일 시간대별 배차간격 및 혼잡도(예: 1호선, 청량리역)

  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/e344f1de-aa4a-4ed9-9a8c-2b4246914b81)  <br>
  
  - 평일 출퇴근 시간대인 7-8시간대, 8-9시간대, 18-19시간대, 19-20시간대에 배차간격이 짧으며, 자주 배차됨에도 불구하고 혼잡도가 높게 나타났다.
  - 출근 시간대인 7-8시간대, 8-9시간대에는 주거지역이 다수 분포하는 청량리역에서 업무지역이 다수 분포하는 서울역으로 가는 방향인 하선의 혼잡도가 높게 나타났다.
  - 반대로 퇴근 시간대인 18-19 시간대, 19-20 시간대에는 반대의 이유로 상선의 혼잡도가 높게 나타났다. 

* 주말 시간대별 배차간격 및 혼잡도(예: 1호선, 청량리역)

  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/30d99185-c828-44d8-bc93-3ab4bb68023c) <br>
  
  - 주말에는 전동차 배차간격이 5분으로 일정했으며, 평일에 비해 혼잡도의 변화가 크지 않았다. 다만, 심야시간대인 22-23시간대, 23-24시간대, 24시 이후 시간대에 혼잡도가 급격히 감소하는 것을 알 수 있다.
 
<br>

## III. Methodology
### Long Short Term Memory (LSTM)
LSTM(Long Short Term Memory) 모델은 기존 RNN(Recurrent Neural Network)의 기울기 소실 문제를 해결하기 위해 개발되었다. LSTM은 RNN의 기본 구조에 셀 상태(Cell state)와 세 가지 게이트를 추가한 구조를 가지고 있다. 이 세 가지 게이트는 Forget Gate, Input Gate, Output Gate로 구성된다. 

<br>

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/47e8106c-67c0-4efe-ad88-63dad6de2380)

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/214e0a58-2108-4ea6-989b-59a44e16f966)

<br>

세 가지 게이트는 다음과 같은 역할을 한다:
- **Forget Gate (망각 게이트):** 과거의 불필요한 정보를 잊도록 결정
- **Input Gate (입력 게이트):** 현재의 정보를 기억하도록 결정
- **Output Gate (출력 게이트):** 어떤 정보를 출력할지 결정



LSTM 네트워크는 셀 상태와 은닉 상태를 통해 다음과 같은 방식으로 정보를 업데이트한다:

1. **셀 상태(Cell State):** 긴 시간 동안 정보를 유지하는 역할을 하며, 많은 시점에 걸쳐 정보를 전달
2. **은닉 상태(Hidden State):** 단기적인 정보를 제공하며, 현재 입력과 셀 상태를 기반으로 매 시점마다 업데이트

LSTM의 업데이트 과정은 다음과 같다:

- **망각 게이트(Forget Gate):** 망각 게이트는 셀 상태에서 어떤 정보를 버릴지 결정
  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/496e8b6c-95e3-40f4-b7bf-419dc1e9fafa)

<br>

- **입력 게이트(Input Gate):** 입력 게이트는 현재 입력에서 어떤 정보를 셀 상태에 추가할지 결정
  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/85b512dd-97d8-44c0-b055-a5cf85437a43)

<br>

- **셀 상태 업데이트(Cell State Update):** 셀 상태는 망각 게이트로 조절된 이전 셀 상태와 입력 게이트로 조절된 새로운 후보 값을 결합하여 업데이트
  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/4561cfa7-73c6-4cbb-b3bc-f83551f09031)
  
  <br>

- **출력 게이트(Output Gate):** 출력 게이트는 셀 상태에 tanh 활성화를 적용하고, 이를 출력 게이트로 조절하여 은닉 상태를 결정
  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/21199fca-2808-4726-a2af-0ed6dd4d9b41)

<br>


마지막으로, 시점 _T_ 에서의 예측값은 다음과 같이 계산된다:
<br>

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/44382d50-a8b3-47c0-8729-4764c38d8c37)

<br>


이와 같은 게이트 구조를 통해 LSTM은 긴 시퀀스에서도 중요한 정보를 효과적으로 유지하고 불필요한 정보를 제거할 수 있다. 또, 셀 상태를 통해 종종 긴 시간에 걸친 패턴과 추세를 포함하는 시계열 데이터를 학습하고 유지할 수 있다.
<br>

따라서 LSTM이 시퀀스 데이터를 다루는 작업에 매우 적합하다고 생각했고, 지하철역이 순차적으로 존재하는 것을 시계열로 해석할 수 있다고 보아, 예측 모델로 LSTM을 선정하였다.

## IV. Evaluation & Analysis
1. 필요한 라이브러리 가져오기 및 GPU/CPU 디바이스 설정
```python
import numpy as np
import pandas as pd
import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)
#check gpu device (if using gpu)
print(torch.cuda.get_device_name(0))
```
<br>

2. 역번호가 199보다 작거나 1000보다 큰 역의 데이터를 제거하여 데이터프레임 재구성
* 역 개수가 많지 않아 학습 데이터로 사용하기 힘든 서울 지하철 1호선(서울역-청량리역) 구간 및 2호선 신정지선, 성수지선을 제외한다. 또 분기가 있는 5호선 강동역 동쪽 구간도 제외하고, 방화-강동 구간의 데이터만을 이용한다.

```python
df = pd.read_csv("2022_final.csv", encoding='cp949')
stations_to_remove = df[(df['역번호'] > 1000) | (df['역번호'] < 199)].index
df.drop(stations_to_remove, inplace=True)
```
<br>

3. MinMaxScaler 이용하여 데이터 스케일링하기
* MinMaxScaler import 및 각각의 스케일러 만들기
  - 데이터 예측 시 특정 feature에 과도하게 영향을 받지 않기 위해 정규화 작업 실행
  - 각 target feature에 대해 MinMaxScaler 만들어 최솟값 0, 최댓값 1로 설정
  - 예측 이후 다시 기존 값으로 변환하기 용이하게 각각 scaler를 만들었음
```python
from sklearn.preprocessing import MinMaxScaler

feature_scaler = MinMaxScaler(feature_range=(0, 1))
up_weekday_scaler = MinMaxScaler(feature_range=(0, 1))
up_saturday_scaler = MinMaxScaler(feature_range=(0, 1))
up_sunday_scaler = MinMaxScaler(feature_range=(0, 1))
down_weekday_scaler = MinMaxScaler(feature_range=(0, 1))
down_saturday_scaler = MinMaxScaler(feature_range=(0, 1))
down_sunday_scaler = MinMaxScaler(feature_range=(0, 1))
```

* 평일, 토요일, 일요일 상/하선 데이터들을 각각의 스케일러를 사용하여 스케일링하고 csv파일로 저장하기
* List of Features :
  - 승차_Weekday/Saturday/Sunday: 각 요일(평일, 토요일, 일요일)의 해당 노선, 해당 역, 해당 시간대에 승차한 인원을 나타낸다.
  - 하차_Weekday/Saturday/Sunday: 각 요일의 해당 노선, 해당 역, 해당 시간대에 하차한 인원을 나타낸다.
  - 환승_Weekday/Saturday/Sunday: 각 요일의 해당 노선, 해당 역, 해당 시간대의 환승 인원 수를 나타낸다.
  - interval_Weekday/Saturday/Sunday: 각 요일, 시간대별 배차 간격을 나타낸다.
  - capacity: 전동차 한 편성의 수용 인원을 나타낸다. (참고: 1-4호선은 10량 1편성, 5-7호선은 8량 1편성, 8호선은 6량 1편성)
  - progression: 각 노선의 기/종점에 가까워질수록 한쪽 방향의 승객 수가 더 많다. 각 노선 끝부분의 예측 정확도를 위해 새롭게 만든 feature이다. 열차가 기점에서 종점까지 운행할 때 어느 정도 운행했는지를 나타내는 수치이다. 
  
```python
features = ['승차_Weekday', '승차_Saturday', '승차_Sunday', '하차_Weekday', '하차_Saturday', '하차_Sunday', '환승_Weekday', '환승_Saturday', '환승_Sunday', 'interval_Weekday', 'interval_Saturday', 'interval_Sunday', 'capacity']

df[features] = feature_scaler.fit_transform(df[features])
df['상선_Weekday'] = up_weekday_scaler.fit_transform(df['상선_Weekday'].to_frame())
df['상선_Saturday'] = up_saturday_scaler.fit_transform(df['상선_Saturday'].to_frame())
df['상선_Sunday'] = up_sunday_scaler.fit_transform(df['상선_Sunday'].to_frame())
df['하선_Weekday'] = down_weekday_scaler.fit_transform(df['하선_Weekday'].to_frame())
df['하선_Saturday'] = down_saturday_scaler.fit_transform(df['하선_Saturday'].to_frame())
df['하선_Sunday'] = down_sunday_scaler.fit_transform(df['하선_Sunday'].to_frame())

df.to_csv('2022_scaled.csv', index=False, encoding='cp949')
```

* 데이터프레임에 'progression'열 생성 및 각 리스트 정의하기(평일)
```python
df['progression'] = [0.0] * len(df)
weekday_up = ['역번호', '승차_Weekday', '하차_Weekday', '환승_Weekday', 'interval_Weekday', 'capacity', 'progression', '상선_Weekday']
weekday_down = ['역번호', '승차_Weekday', '하차_Weekday', '환승_Weekday', 'interval_Weekday', 'capacity', 'progression', '하선_Weekday']
weekday_up2 = ['승차_Weekday', '하차_Weekday', '환승_Weekday', 'interval_Weekday', 'capacity', 'progression', '상선_Weekday']
weekday_down2 = ['승차_Weekday', '하차_Weekday', '환승_Weekday', 'interval_Weekday', 'capacity', 'progression', '하선_Weekday']
hours = ['06-07시간대', '07-08시간대', '08-09시간대', '09-10시간대', '10-11시간대', '11-12시간대', '12-13시간대', '13-14시간대', '14-15시간대', '15-16시간대', '16-17시간대', '17-18시간대', '18-19시간대', '19-20시간대', '20-21시간대', '21-22시간대', '22-23시간대', '23-24시간대']
```

* 각 노선의 평일 상/하행 데이터를 시간대별로 분리하여 정리한 후, 해당 데이터를 csv파일로 저장한다.
* 2호선의 경우 순환열차이기 때문에 progression 값을 0.5로 설정하고 이외의 노선은 역번호와 시작 번호를 기반으로 progression 값을 계산한다.
* 각 csv 파일은 “노선_시간대_상선/하선” 형식의 파일명을 가지며, 각 row는 특정 역의 feature들을 가지고 있다. 역의 배열 순서는 상선 데이터일 경우 역번호 내림차순, 하선 데이터일 경우 역번호 오름차순이며 각 csv 파일이 노선 운행 기점부터 종점까지의 하나의 시계열이 된다. 이렇게 추출한 많은 시계열 데이터를 토대로 학습을 진행한다.

```python
start = [0, 0, 0, 9, 5, 10, 10, 9, 10]
end = [0, 0, 0, 52, 34, 48, 47, 50, 27]
num_stations = [0, 0, 43, 44, 51, 56, 39, 53, 18]

for line in range(2,9):
    for period in hours:
        tmp = df.loc[df['호선'] == line]
        tmp = tmp[weekday_up]
        tmp2 = tmp.loc[df['hour'] == period]
        tmp2 = tmp2.sort_values(by='역번호', axis=0, ascending=False, inplace=False)
        if (line == 2):
            tmp2['progression'] = [0.5] * len(tmp2)
        else:
            tmp2['progression'] = (num_stations[line] - (tmp2['역번호'] - line * 100 - start[line])) / num_stations[line]
        tmp3 = tmp2[weekday_up2]
        pd.DataFrame(tmp3).to_csv(f'weekday_split\\{line}_{period}_up.csv', index=False, encoding='cp949')

    for period in hours:
        tmp = df.loc[df['호선'] == line]
        tmp = tmp[weekday_down]
        tmp2 = tmp.loc[df['hour'] == period]
        tmp2 = tmp2.sort_values(by='역번호', axis=0, ascending=True, inplace=False)
        if (line == 2):
            tmp2['progression'] = [0.5] * len(tmp2)
        else:
            tmp2['progression'] = (tmp2['역번호'] - line * 100 - start[line]) / num_stations[line]
        tmp3 = tmp2[weekday_down2]
        pd.DataFrame(tmp3).to_csv(f'weekday_split\\{line}_{period}_down.csv', index=False, encoding='cp949')
```

* 토요일, 일요일의 데이터도 위의 코드와 마찬가지로 처리한다.

<br>

4. 모델 학습에 사용할 수 있도록 데이터셋 준비하기

* 데이터를 저장할 리스트를 생성하고 시계열 길이를 8로 설정하기.
* 각 디렉토리 내의 csv파일을 읽고, 데이터를 PyTorch 텐서로 변환하기
```python
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Assume your time series data is already preprocessed and in the format of PyTorch tensors
# Each time series is a 2D tensor of shape (sequence_length, num_features)

# Training data (multiple time series)
import os
# assign directory
directories = ['weekday_split', 'saturday_split', 'sunday_split']
time_series_list = []
X = []
y = []
time_steps = 8

for directory in directories:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            tmp = pd.read_csv(f, encoding='cp949')
            tmp = tmp.astype(float)
            tens = torch.from_numpy(tmp.values)
            time_series_list.append(tens)
```

* 각 시계열 데이터를 슬라이딩 윈도우 방식으로 나눠 입력 시퀀스 'x'와 타겟 값 'y' 생성하기
* 타겟 값 'y'를 배열 형식으로 변환하고, 'x'와 'y'를 PyTorch 텐서로 변환하고 데이터 형식을 'float32'로 설정하기
* 데이터 형태 확인
```python
for ts in time_series_list:
    for i in range(len(ts) - time_steps):
        X.append(ts[i:i + time_steps])
        y.append(ts[i + time_steps, -1])  # Assuming the last feature is the target

y = np.array(y)
# Convert to PyTorch tensors
X = torch.stack(X, dim=0)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
print(X.size())
print(y.size())
```

* 'Dataset' class를 상속하여 'TimeSeriesDataset' 클래스 정의하기
* 'x'와 'y' 데이터를 사용하여 'TimeSeriesDataset' 인스턴스 생성
* 전체 데이터셋을 학습용 데이터셋(80%)과 검증용 데이터셋(20%)으로 분할하기
* 학습용, 검증용 데이터셋을 위한 데이터 로더 생성
```python
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].unsqueeze(-1)

dataset = TimeSeriesDataset(X, y)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% of the data for training
val_size = len(dataset) - train_size  # Remaining 20% for validation

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```
<br>

5. LSTM 모델 정의하고, cost function과 최적화 알고리즘 설정하기
* 'nn.Module' 상속하여 'LSTMModel' 클래스 정의하기
* cost function으로는 Mean-Squared-Error 사용
* 최적화 알고리즘으로 Adam optimizer 사용하며, learning rate는 0.001로 설정
```python
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

input_size = X.shape[2]  # Number of features in the input data
hidden_size = 60        # Number of features in the hidden state
num_layers = 4        # Number of stacked LSTM layers
output_size = 1          # Number of output features (1 target feature)

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
<br>

6. 데이터 학습 및 검증 절차 수행하기
* 학습 epoch 수는 70으로 설정
* 각 epoch마다 학습과 검증을 반복함
```python
# Training loop
num_epochs = 70
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
```
```
Epoch [1/70], Loss: 0.0024, Val Loss: 0.0025
Epoch [2/70], Loss: 0.0106, Val Loss: 0.0023
Epoch [3/70], Loss: 0.0009, Val Loss: 0.0023
Epoch [4/70], Loss: 0.0021, Val Loss: 0.0020
Epoch [5/70], Loss: 0.0026, Val Loss: 0.0020
Epoch [6/70], Loss: 0.0017, Val Loss: 0.0019
Epoch [7/70], Loss: 0.0021, Val Loss: 0.0025
Epoch [8/70], Loss: 0.0026, Val Loss: 0.0021
Epoch [9/70], Loss: 0.0021, Val Loss: 0.0022
Epoch [10/70], Loss: 0.0014, Val Loss: 0.0019
Epoch [11/70], Loss: 0.0011, Val Loss: 0.0020
Epoch [12/70], Loss: 0.0016, Val Loss: 0.0017
Epoch [13/70], Loss: 0.0012, Val Loss: 0.0018
Epoch [14/70], Loss: 0.0027, Val Loss: 0.0022
Epoch [15/70], Loss: 0.0023, Val Loss: 0.0017
<중략>
Epoch [60/70], Loss: 0.0005, Val Loss: 0.0008
Epoch [61/70], Loss: 0.0006, Val Loss: 0.0008
Epoch [62/70], Loss: 0.0004, Val Loss: 0.0009
Epoch [63/70], Loss: 0.0004, Val Loss: 0.0008
Epoch [64/70], Loss: 0.0004, Val Loss: 0.0008
Epoch [65/70], Loss: 0.0002, Val Loss: 0.0008
Epoch [66/70], Loss: 0.0004, Val Loss: 0.0008
Epoch [67/70], Loss: 0.0004, Val Loss: 0.0008
Epoch [68/70], Loss: 0.0004, Val Loss: 0.0008
Epoch [69/70], Loss: 0.0008, Val Loss: 0.0008
Epoch [70/70], Loss: 0.0005, Val Loss: 0.0008
```
* 결과값을 보면, 학습 과정에서 모델이 안정적으로 개선되고 있으며 최종 epoch에서 Loss: 0.0005, Val Loss: 0.0008로 검증데이터에서도 손실값이 낮게 나왔다. 이는 모델이 과적합되지 않고 일반화 성능도 높게 나타난다고 볼 수 있다.

<br>

7. 모델 사용하여 예측 수행하기
* test를 위해 새로운 시계열 데이터를 사용하여 LSTM 모델로 예측을 수행하고, 예측된 값과 실제 값 출력하기

```python
scaled_new_ts = pd.read_csv("4_20-21시간대_up.csv", encoding='cp949')
scaled_new_ts = scaled_new_ts.to_numpy()
print(scaled_new_ts.shape)
```

```python
# New time series for testing
actual = scaled_new_ts[:, -1].copy()
scaled_new_ts[time_steps:, -1] = 0.0
X_test = scaled_new_ts.copy()
# Initialize the placeholder for predictions
predictions = []
```

```python
# Make predictions
model.eval()
with torch.no_grad():
    for t in range(time_steps, X_test.shape[0]+1):
        # Prepare the input for the model
        X_input = X_test[t-time_steps:t, :]  # Inputs up to the current time step
        X_input = [X_input]
        
        # Convert to tensor
        X_input_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
        
        # Predict the target feature
        y_pred = model(X_input_tensor)
        pred_value = y_pred.cpu().numpy()[0][0]
        # Update the placeholder with the predictions
        predictions.append(pred_value)
        
        # Update the input with the predicted target feature for the next step
        if (t != X_test.shape[0]):
            X_test[t, -1] = pred_value
        
predicted = predictions
actual = actual[time_steps:]
#predicted = np.reshape(predicted, (-1,1))
#predicted = down_saturday_scaler.inverse_transform(predicted)  #switch scaler as needed
#actual = np.reshape(actual, (-1,1))
#actual = down_saturday_scaler.inverse_transform(actual)
    
# Print the final prediction values of the last feature (target feature)
print("Predicted values:", predicted)  # Remove the extra dimension for readability
print("Actual values:", actual)
```
```
Predicted values: [0.49223268, 0.55302805, 0.59948254, 0.6101891, 0.68271565, 0.50707304, 0.47681022, 0.44612676, 0.42431593, 0.33253032, 0.33147523, 0.3468335, 0.30739588, 0.37064248, 0.3793793, 0.36240828, 0.75844723, 0.6545705]
Actual values: [0.46213808 0.51670379 0.53619154 0.58296214 0.58685969 0.45712695
 0.39587973 0.33853007 0.30289532 0.21046771 0.19988864 0.20824053
 0.20211581 0.22884187 0.2188196  0.22104677 0.3435412 ]
```

* 예측 값과 실제 값 사이에 약간의 오차가 존재함을 확인할 수 있었다.

<br>

## V. Related Work (e.g., existing studies)
* Time Series Forecasting using Pytorch
  - https://www.geeksforgeeks.org/time-series-forecasting-using-pytorch/
* Multivariate Time Series Forecasting Using LSTM
  - https://medium.com/@786sksujanislam786/multivariate-time-series-forecasting-using-lstm-4f8a9d32a509
<br>

## VI. Conclusion: Discussion

### Conclusion

test 데이터인 2021년도의 데이터 중 2개의 데이터를 가져와 MSE 값과 R-squared score를 확인해보았다. <br>
(hidden_size = 70  num_layers = 5 epochs = 50) <br>
MSE는 값이 작을수록 예측이 정확하며, R-squared score는 0과 1 사이의 값을 가지는데, 1에 가까울수록 모델의 데이터 설명력이 좋은 것이다.
<br>

* 일요일 3_17~18 시간대(상선)
  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/228aed37-9cbe-4209-b968-f520c9a60b64)

  ```
  MSE: 21.9564
  R^2 Score: 0.3985
  ```
  - MSE가 21.9564라는 것은 예측값과 실제값 사이의 오차가 평균적으로 크지 않다고 볼 수 있다.
  - R-squared score을 바탕으로 보면 이 모델은 데이터의 변동성의 약 39.85%를 설명한다. 이는 모델의 설명력이 조금 떨어짐을 나타낸다.
<br>

* 일요일 5_12~13 시간대(하선)

  ![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/c9ae68fa-7f71-4c54-b640-cd7afd57d4bd)

  ```
  MSE: 39.9610
  R^2 Score: 0.4485
  ```
  - MSE가 39.9610라는 것은 앞서 예측한 데이터보다는 예측값과 실제값 사이의 오차가 커졌다고 볼 수 있다.
  - R-squared score을 바탕으로 보면 이 모델은 데이터의 변동성의 약 44.85% 를 설명한다. 이를 바탕으로 모델의 설명력이 앞서 예측한 것보다 더 낫다는 것을 알 수 있다.

<br>

### Discussion

지하철 데이터들이 환승역의 데이터가 한 쪽 노선으로 편중되어있다는 점, 기/종점에 가까운 역의 승차 인원은 한 쪽 방향으로 더 많이 몰려 있다는 점과 같이, 데이터를 다룰 때 실제와 예측을 비슷하게 하기 위해 전처리해야 하는 과정이 까다롭고 길었다. 
<br>
LSTM 기반 예측 모델을 통한 학습이 매우 잘 되었고, 예측 결과도 실제값과 오차가 작은 편이라는 점이 긍정적이다. 하지만 모델의 설명력이 부족한 부분은 보완해야 할 것이다. 하이퍼파라미터를 조정하여 정확도를 더 높일 수 있을 것 같다.
<br>
또한, time_steps = 8 에서 이전 8개의 역 데이터를 보고 모델이 다음 값을 예측하는데, 이 프로젝트에서는 시간 관계상 테스트 과정에서 첫 8개의 역 데이터를 예측하지 않고 직접 정답 데이터를 주었다. 기점 부분의 역들은 이전 역 데이터의 영향을 많이 받지 않으니 random forest와 같은 회귀 모델로 먼저 예측을 한 후 lstm 모델을 돌려보면 완전한 혼잡도 예측이 가능할 것 같다. 
<br>
딥러닝 기술을 통해 지하철 혼잡도 예측 기술을 더욱 향상시키고 고도화한다면, 유동인구가 많은 수도권의 대중교통 이용자들의 안전과 교통관리 및 편의서비스를 제공할 수 있을 것으로 보인다.

<br>

## VII. Credits
* 강민성 : data preprocessing, code implementation
* 김승윤 : data visualization, methodology introduction
* 오세원 : YouTube recording
* 황윤영 : write up Github, make conclusion
