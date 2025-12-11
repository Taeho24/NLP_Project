# NLP_Project - Game Tag Summary and Recommendation

## 프로젝트 목표

이 프로젝트는 **게임 리뷰 데이터**를 기반으로 사용자 성향을 분석하고, 이를 반영하여 게임의 **요약 평가**와 **새로운 태그**를 자동 생성하는 **Review-Based Game Summary & Tag Recommendation 모델**을 제안합니다.

### 핵심 문제 정의 및 목표

| 구분 | 내용 |
| --- | --- |
| **문제 정의** | 기존 게임 태그가 개발사의 의도나 마케팅 전략에 따라 설정되어 **실제 이용자 인식과 괴리가 발생하는 문제** 해결. |
| **연구 목표** | 사용자 리뷰 경험을 기반으로 게임 태그를 재정의하고, **사용자 경험 중심의 동적 태그 생태계** 구축 가능성을 제시. |
| **주요 기능** | 사용자의 리뷰 이력과 문체를 분석하여 **성향 벡터(Persona Vector)**를 도출하고, 이 성향에 따라 리뷰 텍스트의 의미를 해석하여 요약 및 태그 생성. |
| **기대 효과** | 개발사 중심의 고정적 분류 체계를 보완하고, **개인화된 게임 평가 및 추천 시스템**의 기반 마련. |

### 도출되는 성향 벡터 축

사용자 성향은 네 가지 축으로 분석됩니다.

* **서사 중심 (Narrative):** 스토리, 서사, 엔딩, 캐릭터 몰입도 선호
* **자유도 중심 (Freedom):** 오픈월드, 탐험, 상호작용, 선택의 자유도 선호
* **안정성 중심 (Stability):** 최적화, 버그, 프레임, 서버 안정성 선호
* **도전 중심 (Challenge):** 난이도, 컨트롤, 보스 패턴, 피지컬 요구 선호

## 아키텍처 및 기술 스택

### 파이프라인 아키텍처 

프로젝트는 다음 3단계의 모듈화된 파이프라인으로 구성되어 있습니다.

1.  **데이터 수집 (Collector):** Steam API 및 웹 크롤링을 통해 게임 리뷰 데이터를 수집하고 원본 JSON 파일로 저장합니다.
2.  **분석 및 전처리 (Analyzer):** 수집된 리뷰 텍스트에서 키워드 기반 성향 벡터를 계산하고, 분석된 중간 결과(CSV)를 생성합니다.
3.  **생성 및 요약 (Generator):** KoBERT 모델을 활용하여 리뷰 텍스트의 요약 문장을 추출하고, 성향 벡터를 기반으로 최종 추천 태그와 분석 결과를 생성합니다.

### 사용 기술 (Tech Stack)

| 구분 | 기술 | 역할 |
| :--- | :--- | :--- |
| **언어/환경** | Python 3.9+ | 전체 개발 환경 및 스크립팅 |
| **프론트엔드** | Streamlit | 웹 기반 분석 대시보드 및 UI 구성 |
| **NLP 모델** | **KoBERT** (SKT KoBERT-Base) | 리뷰 텍스트의 문장 임베딩 및 요약 추출 |
| **데이터 처리**| Pandas, NumPy, Scikit-learn | 성향 벡터 계산, 집계, K-Means 클러스터링 기반 요약 |
| **데이터 수집** | `steamreviews`, `requests`, `BeautifulSoup` | Steam 게임 리뷰 데이터 크롤링 |
| **딥러닝** | PyTorch, Hugging Face Transformers | KoBERT 모델 구동 및 관리 |

## 📁 디렉토리 구조

```
NLP_PROJECT/
├── app.py                      \# Streamlit 메인 실행 파일 (파이프라인 컨트롤러)
├── data/                       \# steamreviews의 크롤링 캐싱 데이터 (.json)
├── dataSet/                    \# 모든 입력/출력 데이터 저장소 (.json, .csv, .txt)
├── collectData/                \# 데이터 수집 모듈 패키지
│   └── collector.py            \# Steam 크롤링 로직
└── model/                      \# 분석 및 생성 모듈 패키지
    ├── analyzer.py             \# 성향 벡터 계산 로직
    └── generator_bert.py       \# KoBERT 모델 로드 및 요약/태그 생성 로직
````

## 설치 및 실행 방법

### 1. 환경 설정

프로젝트를 클론하고 가상 환경을 활성화합니다.

```bash
# 프로젝트 클론
git clone [Repository URL]
cd NLP_Project

# 가상 환경 생성 및 활성화 (Python 3.9 이상 권장)
python -m venv venv
source venv/bin/activate  # Linux/macOS
source .\venv\Scripts\activate   # Windows
````

### 2. 의존성 설치

필요한 모든 라이브러리를 설치합니다. (KoBERT 및 PyTorch 포함)

```bash
# PyTorch 설치 (자신의 환경에 맞는 버전을 설치하세요. CPU 버전 예시)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 나머지 라이브러리 설치
pip install -r requirements.txt
```

### 3. 디렉토리 준비

결과를 저장할 디렉토리를 생성합니다.

```bash
mkdir dataSet
```

### 4\. Streamlit 앱 실행

**프로젝트의 루트 디렉토리**에서 `app.py`를 실행합니다.

```bash
streamlit run app.py
```

브라우저가 열리면, UI에서 분석하고자 하는 게임 이름을 입력하고 파이프라인을 실행할 수 있습니다. (게임 이름은 영어명 인식을 권장합니다.)

```