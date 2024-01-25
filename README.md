[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/g6ZC_OOE)
# FastCampus AI Lab ML 프로젝트 - 5조

## Team

| ![박패캠](https://tvstore-phinf.pstatic.net/20210907_263/1631002069199vDKNA_JPEG/00033.jpg) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![윤수인](https://drive.google.com/file/d/11bjWNo5S5Yyqs84BUbq2RFrfTQ5E7Cox/view?usp=drive_link) | 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [윤수인](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            EDA, 성능 실험                             |

## 1. Competiton Info

### Overview

- AiStages 아파트 실거래가 예측
	> 서울시 아파트 실거래가 매매 데이터를 기반으로 아파트 가격을 예측하는 대회

### Timeline

- 2024.01.15 - Start Date
- ~ 2024.01.25 19:00 - Final submission deadline

### Evaluation

- 매매 실거래가를 예측하는 Regression 대회이며, 평가지표는 RMSE(Root Mean Squared Error)를 사용합니다.

## 2. Components

### Directory

- final-code
	- code
		- final_code.ipynb - 최종 파일
	- apartments.csv - 서울시 공동주택 아파트 데이터
	- coordinates.csv - 위/경도 데이터
	- subway_counts.csv - 500m 이내 지하철역 개수 데이터 
	- top_apt_coords.csv - 상위 아파트가 속한 동 및 대표 아파트 위/경도 데이터

## 3. Data descrption

### Dataset overview

- 학습 데이터: (1,118,822, 52)
	- 예측해야 할 거래금액(target)을 포함하여 아파트의 정보에 대한 52개의 변수와 거래시점에 대한 변수가 주어집니다.
	- 2007년 1월 1일부터 2023년 6월 30일까지의 거래 데이터로 이루어져 있습니다.
- 평가 데이터: (9272, 51)
	- 2023년 7월 1일부터 2023년 9월 26일까지의 거래 데이터로 이루어져 있으며, 거래금액(target)을 제외한 51개의 변수가 주어집니다.

### EDA

- 변수별 분포	
	![output](https://github.com/UpstageAILab/upstage-ml-regression-05/assets/40953615/78f35cda-723f-4832-9113-5a3e6352b924)
	- 0의 개수가 10000개 이상인 변수: 'k-전용면적별세대_60', 'k-전용면적별세대_60_85', 'k-전용면적별세대_85_135', '건축면적', '주차대수'
		- k-전용면적별세대_60, k-전용면적별세대_60_85, k-전용면적별세대_85_135의 경우, 실제 값이 0인 것으로 보입니다.
		- 건축면적, 주차대수의 경우, 0이 결측치일 가능성이 있습니다. 만약 결측치라면 아파트 단지 정보이므로 해당 아파트의 실제 해당 정보 값으로 대체가 가능합니다.
- 상관관계 분석
	![output01](https://github.com/UpstageAILab/upstage-ml-regression-05/assets/40953615/248d0600-5282-4e0f-b211-3d3434a803f3)
	- k-전체동수, k-전체세대수, k-연면적, k-주거전용면적, k-관리비부과면적, k-전용면적별세대, 주차대수 변수 간에 높은 상관관계를 보입니다.
	- 아파트 단지 관련 변수들 간에 높은 양의 상관관계를 보입니다.
	- 건축면적은 0으로 되어 있는 17399개의 결측치를 대체하면 다른 변수들과의 상관계수 달라질 것으로 예상됩니다.
	- target과의 상관관계가 높은 변수는 전용면적, 주차대수, 계약년월, k-연면적, k-주거전용면적, k-관리비부과면적, 좌표Y입니다.
		- 전용면적: 계약된 가구의 전용면적
		- 주차대수: 아파트 단지 전체의 주차 가능 대수
		- 계약년월과 양의 상관관계, 최근일수록 실거래가 높음
		- k-연면적: 아파트 단지 전체의 연면적, k-주거전용면적과의 상관계수 0.90
		- k-주거전용면적: 아파트 단지 전체 전용면적 = SUM(전용면적별세대수 x 전용면적)
		- k-관리비부과면적: k-주거전용면적 or k-연면적
		- 좌표Y(위도)와 음의 상관관계, 남쪽으로 갈수록 실거래가 높음
	-  결측치 확인
		![output02](https://github.com/UpstageAILab/upstage-ml-regression-05/assets/40953615/85a79df0-a155-455b-8c28-8bfbffce2646)
		- 일부 결측치는 서울시 공동주택 아파트 정보(https://data.seoul.go.kr/dataList/OA-15818/A/1/datasetView.do )를 사용해 대체할 수 있었습니다.
		- 나머지 결측치 처리 방법
			- 결측치가 50만개 이상인 변수
				-  '주차대수'를 제외하고 모두 제거하였습니다.
			- 수치형 변수
				- '주차대수': 회귀 모델(RandomForestRegressor)을 사용하여 대체하였습니다.
				- geopy 라이브러리를 사용해 도로명 주소를 좌표로 변환하여 채워주었습니다.
			- 범주형 변수
				- 'NULL'이라는 임의의 범주로 대체하였습니다.
	- 이상치 처리
   		- 'target'(실거래가) 기준으로 IQR을 이용해 아웃라이어인 데이터를 이상치로 판단하여 제거해보았습니다.
		- 최종 제출 파일로 이상치를 제거한 코드와 제거하지 않은 코드의 예측 파일 2개를 선정하였습니다.

### Feature engineering
- 파생변수 생성
	- 시군구 컬럼을 '시'와 '군'으로 분할
	- '계약년월'을 '계약년', '계약월'로 분할
	- '도로명'(전체 도로명 주소)에서 '도로'(도로 이름, 예: 삼성로)만 추출
	- '부촌여부' 변수 추가: 실거래라 상위 아파트가 많이 위치한 동(청담동, 한남동, 성수동 1가) 여부
	- '상위아파트여부' 변수 추가: 실거래가 top10 아파트 여부
	- '대장아파트거리' 변수 추가: 지역구별 대장 아파트 기준 거리
	- 'top아파트거리' 변수 추가: 동별 상위 아파트와의 거리 
	- '건물연식' 변수 추가: 계약년 - 건축년도
	- '브랜드명' 변수 추가: '아파트명'이 주요 브랜드명을 포함하는 경우 해당 브랜드명 입력
	- '인근지하철역개수' 변수 추가: 주어진 subway_feature 데이터와 좌표 변수를 이용해 인근 500m 내에 위치한 지하철역의 개수 입력
   	- '인근버스정류장개수' 변수 추가: 주어진 bus_feature 데이터를 좌표 변수를 이용해 인근 버스정류장역 개수 입력
- 그 외 외부 데이터를 사용한 파생변수
  	- '기준금리'
  	- '전세가격지수'
  	- '인구밀도'
  	- '인근종합공원개수'
  	- '인근병원개수'
  	- '인근학교개수'
  	- '한강지천생활지수'
- 타겟 인코딩
	- 일부 범주형 변수(도로명, 도로, 동)에 대해 실거래가 평균 기준 순위로 인코딩하였습니다. -> '도로명_실거래가순위', '동_실거래가순위', '도로_실거래가순위'
- Feature Selection
	- 주어진 데이터의 변수들 중 필요 없다고 판단되는 변수, A/B Test 결과 성능 향상에 도움이 되지 않는 변수, 모델 학습 결과 중요도가 낮은 변수를 제거하고 아래의 20개 변수만 사용하였습니다.
		- '도로명_실거래가순위', '전용면적', 'k-복도유형', 'k-단지분류', '계약년', '계약월', '동_실거래가순위', '좌표X', '좌표Y', '건축년도', '부촌여부', '상위아파트여부', '대장아파트거리', '도로_실거래가순위', '구', '주차대수', '인근지하철역개수', '브랜드명', '건물연식', 'top아파트거리'
		
## 4. Modeling

### Model description

- 여러 모델로 학습을 진행해보고 가장 성능이 좋았던 LGBMRegressor를 사용했습니다.
- Validation 방식
	- 초기에는 K-fold Cross Validation으로 학습 성능을 검증하였으나, Test Dataset이 Training Dataset보다 미래의 데이터로 이루어져 있는 특성을 반영하여 신뢰할 수 있는 Validation Set 구축을 위해 '계약년월일'을 기준으로 최근 20%의 거래 데이터를 Validation Set으로 추출하였습니다. 

### Modeling Process
- 업샘플링
	- 모델 학습 결과 Validation 예측값 분석을 통해 주로 실거래가가 높은 데이터의 예측값이 높은 Squared Error를 보인다는 것을 파악하였습니다. 데이터 불균형에 따른 것으로 판단하여 거래가가 높은(100억 이상인) 데이터의 개수를 늘려보았습니다. 
- 하이퍼파라미터 튜닝
	- Optuna를 사용해 최적의 파라미터 조합을 사용해 성능을 평가하였습니다.
	- 이후 직접 값을 변경해가며 가장 좋은 성능을 보이는 값으로 설정하였습니다.

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference

- _Insert related reference_
