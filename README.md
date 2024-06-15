# 지하철 혼잡도 예측해보기
## 목차
1. [Members](#members)
2. [Proposal](#i-proposal)
3. [Datasets](#ii-datasets)
4. [Methodology](#iii-methodology)
5. [Evaluation & Analysis](#iv-evaluation--analysis)
6. [Related Work](#v-related-work-eg-existing-studies)
7. [Conclusion](#vi-conclusion-discussion)

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
* 연신내역의 6호선 승하차량이 모두 3호선의 데이터로 집계되어있어, 3,6호선 각각의 승하차량 비율에 따라 나눈다.
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

* 창동역의 1호선(경원선) 승하차량이 모두 4호선의 데이터로 집계되어있어, 1,4호선 각각의 승하차량 비율에 따라 나눈다.
```python
rate_4 = 1304648 / 3395546
Chang_dong = pivot_df.loc[(pivot_df['역명'] == '창동') & (pivot_df['호선'] == 4)]

for idx, row in Chang_dong.iterrows():
    for col in columns:
        pivot_df.at[idx, col] = row[col] * rate_4
```
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

station_transfer.head
station_transfer.columns
station_transfer.dtypes
```
<br>

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
* 평일 시간대별 승차 인원 및 상/하선 혼잡도(예: 청량리역)
```python
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")

ticklabel=['~6','6~7', '7~8', '8~9', '9~10', '10~11', '11~12', '12~13', '13~14', '14~15', '15~16', '16~17', '17~18', '18~19', '19~20', '20~21', '21~22', '22~23','23~24','24~']


# 특정 역에 대한 데이터 필터링 (예: 청량리 역)
station_data = data[data['역명'] == '청량리']

# 시간대별 승차 인원 시각화 (평일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['승차_Weekday'], label='승차_Weekday')
plt.plot(station_data['hour'], station_data['하차_Weekday'], label='하차_Weekday')
plt.title('승차 및 하차 인원 (평일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('인원수')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()

# 시간대별 상선/하선 혼잡도 시각화 (평일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['상선_Weekday'], label='상선_Weekday')
plt.plot(station_data['hour'], station_data['하선_Weekday'], label='하선_Weekday')
plt.title('상선 및 하선 혼잡도 (평일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()
```
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/0434f090-2952-4cef-b0af-7722dd987526)
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/920af747-bac8-4e4d-88fa-72424c69eb9a)

* 토요일 시간대별 승차 인원 및 상/하선 혼잡도(예: 청량리역)
```python
# 특정 역에 대한 데이터 필터링 (예: 청량리 역)
station_data = data[data['역명'] == '청량리']

# 시간대별 승차 인원 시각화 (토요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['승차_Saturday'], label='승차_Saturday')
plt.plot(station_data['hour'], station_data['하차_Saturday'], label='하차_Saturday')
plt.title('승차 및 하차 인원 (토요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('인원수')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()

# 시간대별 상선/하선 혼잡도 시각화 (토요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['상선_Saturday'], label='상선_Saturday')
plt.plot(station_data['hour'], station_data['하선_Saturday'], label='하선_Saturday')
plt.title('상선 및 하선 혼잡도 (토요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()
```
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/e742c129-9ab1-4da5-8faa-4e06d8182a20)
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/48731efa-e45a-4a3b-a5d0-af57d5ba4b4c)

* 일요일 시간대별 승차 인원 및 상/하선 혼잡도(예: 청량리역)
```python
# 특정 역에 대한 데이터 필터링 (예: 청량리 역)
station_data = data[data['역명'] == '청량리']

# 시간대별 승차 인원 시각화 (일요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['승차_Sunday'], label='승차_Sunday')
plt.plot(station_data['hour'], station_data['하차_Sunday'], label='하차_Sunday')
plt.title('승차 및 하차 인원 (일요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('인원수')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()

# 시간대별 상선/하선 혼잡도 시각화 (일요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['상선_Sunday'], label='상선_Sunday')
plt.plot(station_data['hour'], station_data['하선_Sunday'], label='하선_Sunday')
plt.title('상선 및 하선 혼잡도 (일요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()
```
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/53109601-ac87-485c-9b04-fef127567807)
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/2267267a-98ff-4ddd-b2cc-30946bbac01f)

* 역별 승하차 인원, 역별 상/하선 혼잡도 시각화 (예: 07-08 시간대)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")

# '07-08시간대' 데이터 필터링
df_filtered = data[data['hour'] == '07-08시간대'].copy()

# 역별 승하차 인원 시각화
plt.figure(figsize=(14, 7))

# 승하차 인원 시각화
plt.subplot(1, 2, 1)
sns.barplot(x='역명', y='승차_Weekday', data=df_filtered, color='blue', label='승차_Weekday')
sns.barplot(x='역명', y='하차_Weekday', data=df_filtered, color='red', label='하차_Weekday', alpha=0.7)
plt.title('역별 승하차 인원 (07-08시간대)')
plt.xlabel('역명')
plt.ylabel('인원 수')
plt.legend()

# 역별 상선 및 하선 혼잡도 시각화
plt.subplot(1, 2, 2)
sns.barplot(x='역명', y='상선_Weekday', data=df_filtered, color='blue', label='상선_혼잡도')
sns.barplot(x='역명', y='하선_Weekday', data=df_filtered, color='red', label='하선_혼잡도', alpha=0.7)
plt.title('역별 혼잡도 (07-08시간대)')
plt.xlabel('역명')
plt.ylabel('혼잡도')
plt.legend()

plt.tight_layout()
plt.show()
```

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/755503cc-4e5a-4eaf-b13d-d4fbe9bf65bf)

* 역별 승하차 인원과 역별 상/하선 혼잡도를 한눈에 비교할 수 있도록 한 그래프 안에 시각화한다. (예: 07-08 시간대)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")

# '07-08시간대' 데이터 필터링
df_filtered = data[data['hour'] == '07-08시간대'].copy()

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(14, 7))

# 막대그래프 - 승차 인원
sns.barplot(x='역명', y='승차_Weekday', data=df_filtered, color='blue', label='승차_Weekday', ax=ax1)
sns.barplot(x='역명', y='하차_Weekday', data=df_filtered, color='red', label='하차_Weekday', alpha=0.7, ax=ax1)

# x축 라벨 및 y축 라벨 설정
ax1.set_xlabel('역명')
ax1.set_ylabel('승하차 인원 수')
ax1.legend(loc='upper left')

# 꺾은선 그래프를 위한 두 번째 y축 생성
ax2 = ax1.twinx()
ax2.plot(df_filtered['역명'], df_filtered['상선_Weekday'], color='green', marker='o', linestyle='None', label='상선_혼잡도')
ax2.plot(df_filtered['역명'], df_filtered['하선_Weekday'], color='orange', marker='o', linestyle='None', label='하선_혼잡도')
ax2.set_ylabel('혼잡도')

# 꺾은선 그래프의 범례 설정
ax2.legend(loc='upper right')

# 그래프 제목 설정
plt.title('역별 승하차 인원 및 혼잡도 (07-08시간대)')

# 레이아웃 조정 및 그래프 표시
fig.tight_layout()
plt.show()
```
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/dd36efac-1e0c-4af7-b2bb-2466211c1098)

<br>

2. 각 요일의(평일, 토요일, 일요일) 시간대별 배차간격 및 혼잡도
* 평일 시간대별 배차간격 및 혼잡도(예: 청량리역)
```python
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")

ticklabel=['~6','6~7', '7~8', '8~9', '9~10', '10~11', '11~12', '12~13', '13~14', '14~15', '15~16', '16~17', '17~18', '18~19', '19~20', '20~21', '21~22', '22~23','23~24','24~']


# 특정 역에 대한 데이터 필터링 (예: 청량리 역)
station_data = data[data['역명'] == '청량리']

# 시간대별 배차간격 인원 시각화 (평일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['interval_Weekday'], label='interval_Weekday')
plt.title('배차간격 (평일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('배차간격')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()

# 시간대별 상선/하선 혼잡도 시각화 (평일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['상선_Weekday'], label='상선_Weekday')
plt.plot(station_data['hour'], station_data['하선_Weekday'], label='하선_Weekday')
plt.title('상선 및 하선 혼잡도 (평일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()
```

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/4f8e7f25-8e90-4c30-9fcb-b572d4401d7d)
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/2975fb1d-b11d-4e2f-b994-907ac10de4c5)

* 토요일 시간대별 배차간격 및 혼잡도(예: 청량리역)
```python
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")

ticklabel=['~6','6~7', '7~8', '8~9', '9~10', '10~11', '11~12', '12~13', '13~14', '14~15', '15~16', '16~17', '17~18', '18~19', '19~20', '20~21', '21~22', '22~23','23~24','24~']


# 특정 역에 대한 데이터 필터링 (예: 청량리 역)
station_data = data[data['역명'] == '청량리']

# 시간대별 배차간격 인원 시각화 (토요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['interval_Saturday'], label='interval_Saturday')
plt.title('배차간격 (토요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('배차간격')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()

# 시간대별 상선/하선 혼잡도 시각화 (토요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['상선_Saturday'], label='상선_Saturday')
plt.plot(station_data['hour'], station_data['하선_Saturday'], label='하선_Saturday')
plt.title('상선 및 하선 혼잡도 (토요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()
```

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/6eb678fb-8f30-46ef-86aa-b4f73ceb3090)
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/2b242431-064f-4aba-aa4a-923143e9b087)

* 일요일 시간대별 배차간격 및 혼잡도(예: 청량리역)
```python
import matplotlib.pyplot as plt

plt.rc("font", family="Malgun Gothic")

ticklabel=['~6','6~7', '7~8', '8~9', '9~10', '10~11', '11~12', '12~13', '13~14', '14~15', '15~16', '16~17', '17~18', '18~19', '19~20', '20~21', '21~22', '22~23','23~24','24~']


# 특정 역에 대한 데이터 필터링 (예: 청량리 역)
station_data = data[data['역명'] == '청량리']

# 시간대별 배차간격 인원 시각화 (일요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['interval_Sunday'], label='interval_Sunday')
plt.title('배차간격 (일요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('배차간격')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()

# 시간대별 상선/하선 혼잡도 시각화 (일요일)
plt.figure(figsize=(12, 6))
plt.plot(station_data['hour'], station_data['상선_Sunday'], label='상선_Sunday')
plt.plot(station_data['hour'], station_data['하선_Sunday'], label='하선_Sunday')
plt.title('상선 및 하선 혼잡도 (일요일, 청량리 역)')
plt.xlabel('시간대')
plt.ylabel('혼잡도')
plt.xticks(ticks=station_data['hour'], labels=ticklabel[:len(station_data['hour'])])
plt.legend()
plt.show()
```

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/98ba921d-1bfd-41a6-bb96-041c252e0751)
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/06516f00-0801-4513-9881-d9d8b948475b)

<br>

3. 특정 호선 특정 시간대 역별 승하차 인원 및 혼잡도(예: 1호선, 07-08시간대)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rc("font", family="Malgun Gothic")

# '07-08시간대' 데이터 필터링
line_number = 1  # 원하는 호선 번호를 설정
df_filtered = data[(data['hour'] == '07-08시간대') & (data['호선'] == line_number)].copy()

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(14, 7))

# 막대그래프 - 승차 인원
sns.barplot(x='역명', y='승차_Weekday', data=df_filtered, color='blue', label='승차_Weekday', ax=ax1)
sns.barplot(x='역명', y='하차_Weekday', data=df_filtered, color='red', label='하차_Weekday', alpha=0.7, ax=ax1)

# x축 라벨 및 y축 라벨 설정
ax1.set_xlabel('역명')
ax1.set_ylabel('승하차 인원 수')
ax1.legend(loc='upper left')

# 꺾은선 그래프를 위한 두 번째 y축 생성
ax2 = ax1.twinx()
ax2.plot(df_filtered['역명'], df_filtered['상선_Weekday'], color='green', marker='o', label='상선_혼잡도')
ax2.plot(df_filtered['역명'], df_filtered['하선_Weekday'], color='orange', marker='o', label='하선_혼잡도')
ax2.set_ylabel('혼잡도')

# 꺾은선 그래프의 범례 설정
ax2.legend(loc='upper right')

# 그래프 제목 설정
plt.title(f'{line_number}호선 역별 승하차 인원 및 혼잡도 (07-08시간대)')

# 레이아웃 조정 및 그래프 표시
fig.tight_layout()
plt.show()
```

![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/cf8e9853-a04f-483b-a5ae-d5835353a62b)


## III. Methodology
### LSTM

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
```python
df = pd.read_csv("2022_final.csv", encoding='cp949')
stations_to_remove = df[(df['역번호'] > 1000) | (df['역번호'] < 199)].index
df.drop(stations_to_remove, inplace=True)
```
<br>

3. MinMaxScaler 이용하여 데이터 스케일링하기
* MinMaxScaler import 및 각각의 스케일러 초기화하기
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
* 2호선의 경우 progression 값을 0.5로 설정하고 이외의 노선은 역번호와 시작 번호를 기반으로 progression 값을 계산한다.
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

* 데이터프레임에 'progression'열 생성 및 각 리스트 정의하기(토요일)
```python
saturday_up = ['역번호', '승차_Saturday', '하차_Saturday', '환승_Saturday', 'interval_Saturday', 'capacity', 'progression', '상선_Saturday']
saturday_down = ['역번호', '승차_Saturday', '하차_Saturday', '환승_Saturday', 'interval_Saturday', 'capacity', 'progression', '하선_Saturday']
saturday_up2 = ['승차_Saturday', '하차_Saturday', '환승_Saturday', 'interval_Saturday', 'capacity', 'progression', '상선_Saturday']
saturday_down2 = ['승차_Saturday', '하차_Saturday', '환승_Saturday', 'interval_Saturday', 'capacity', 'progression', '하선_Saturday']
```

* 각 노선의 토요일 상/하행 데이터를 시간대별로 분리하여 정리한 후, 해당 데이터를 csv파일로 저장한다.
* 2호선의 경우 progression 값을 0.5로 설정하고 이외의 노선은 역번호와 시작 번호를 기반으로 progression 값을 계산한다.
```python
for line in range(2,9):
    for period in hours:
        tmp = df.loc[df['호선'] == line]
        tmp = tmp[saturday_up]
        tmp2 = tmp.loc[df['hour'] == period]
        tmp2 = tmp2.sort_values(by='역번호', axis=0, ascending=True, inplace=False)
        if (line == 2):
            tmp2['progression'] = [0.5] * len(tmp2)
        else:
            tmp2['progression'] = (num_stations[line] - (tmp2['역번호'] - line * 100 - start[line])) / num_stations[line]
        tmp3 = tmp2[saturday_up2]
        pd.DataFrame(tmp3).to_csv(f'saturday_split\\{line}_{period}_up.csv', index=False, encoding='cp949')

    for period in hours:
        tmp = df.loc[df['호선'] == line]
        tmp = tmp[saturday_down]
        tmp2 = tmp.loc[df['hour'] == period]
        tmp2 = tmp2.sort_values(by='역번호', axis=0, ascending=False, inplace=False)
        if (line == 2):
            tmp2['progression'] = [0.5] * len(tmp2)
        else:
            tmp2['progression'] = (num_stations[line] - (tmp2['역번호'] - line * 100 - start[line])) / num_stations[line]
        tmp3 = tmp2[saturday_down2]
        pd.DataFrame(tmp3).to_csv(f'saturday_split\\{line}_{period}_down.csv', index=False, encoding='cp949')
```

* 데이터프레임에 'progression'열 생성 및 각 리스트 정의하기(일요일)
```python
sunday_up = ['역번호', '승차_Sunday', '하차_Sunday', '환승_Sunday', 'interval_Sunday', 'capacity', 'progression', '상선_Sunday']
sunday_down = ['역번호', '승차_Sunday', '하차_Sunday', '환승_Sunday', 'interval_Sunday', 'capacity', 'progression', '하선_Sunday']
sunday_up2 = ['승차_Sunday', '하차_Sunday', '환승_Sunday', 'interval_Sunday', 'capacity', 'progression', '상선_Sunday']
sunday_down2 = ['승차_Sunday', '하차_Sunday', '환승_Sunday', 'interval_Sunday', 'capacity', 'progression', '하선_Sunday']
```

* 각 노선의 일요일 상/하행 데이터를 시간대별로 분리하여 정리한 후, 해당 데이터를 csv파일로 저장한다.
* 2호선의 경우 progression 값을 0.5로 설정하고 이외의 노선은 역번호와 시작 번호를 기반으로 progression 값을 계산한다.
```python
for line in range(2, 9):
    for period in hours:
        tmp = df.loc[df['호선'] == line]
        tmp = tmp[sunday_up]
        tmp2 = tmp.loc[df['hour'] == period]
        tmp2 = tmp2.sort_values(by='역번호', axis=0, ascending=True, inplace=False)
        if (line == 2):
            tmp2['progression'] = [0.5] * len(tmp2)
        else:
            tmp2['progression'] = (num_stations[line] - (tmp2['역번호'] - line * 100 - start[line])) / num_stations[line]
        tmp3 = tmp2[sunday_up2]
        pd.DataFrame(tmp3).to_csv(f'sunday_split\\{line}_{period}_up.csv', index=False, encoding='cp949')

    for period in hours:
        tmp = df.loc[df['호선'] == line]
        tmp = tmp[sunday_down]
        tmp2 = tmp.loc[df['hour'] == period]
        tmp2 = tmp2.sort_values(by='역번호', axis=0, ascending=False, inplace=False)
        if (line == 2):
            tmp2['progression'] = [0.5] * len(tmp2)
        else:
            tmp2['progression'] = (num_stations[line] - (tmp2['역번호'] - line * 100 - start[line])) / num_stations[line]
        tmp3 = tmp2[sunday_down2]
        pd.DataFrame(tmp3).to_csv(f'sunday_split\\{line}_{period}_down.csv', index=False, encoding='cp949')
```
<br>

3. 모델 학습에 사용할 수 있도록 데이터셋 준비하기

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
* 'x'와 'y' 데이털르 사용하여 'TimeSeriesDataset' 인스턴스 생성
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

4. LSTM 모델 정의하고, cost function과 최적화 알고리즘 설정하기
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

5. 데이터 학습 및 검증 절차 수행하기
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
<br>

6. 모델 사용하여 예측 수행하기
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
<br>

7. 실제 값과 예측 값 비교 시각화 
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
```
![image](https://github.com/YoonYeongHwang/AIXDeepLearning/assets/170499968/844a0cd3-edaf-42b2-b959-f079b81ed73d)


## V. Related Work (e.g., existing studies)
* Time Series Forecasting using Pytorch
  - https://www.geeksforgeeks.org/time-series-forecasting-using-pytorch/
* Multivariate Time Series Forecasting Using LSTM
  - https://medium.com/@786sksujanislam786/multivariate-time-series-forecasting-using-lstm-4f8a9d32a509

## VI. Conclusion: Discussion
