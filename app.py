# app.py
# 실행방법: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json

# --- 환경 설정 ---
DATA_DIR = "dataSet"
PERSONA_AXES = ['narrative', 'freedom', 'stability', 'challenge']
PERSONA_LABELS_KO = {
    'narrative': '스토리/서사 선호',
    'freedom': '자유도/탐험 선호',
    'stability': '최적화/안정성 선호',
    'challenge': '도전/난이도 선호'
}

# 데이터 로드 및 캐싱
@st.cache_data
def load_data(filename):
    """분석된 CSV 파일과 BERT 분석 텍스트 파일을 로드합니다."""
    
    # 1-1. 분석된 리뷰 데이터 (CSV) 로드
    csv_path = os.path.join(DATA_DIR, f"analyzed_{filename}_reviews.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, None, f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}"

    # 1-2. BERT 분석 결과 (TXT) 로드
    txt_path = os.path.join(DATA_DIR, f"BERT_Analysis_{filename}.txt")
    bert_summary = {}
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 텍스트 파일 내용 파싱 (간단하게 요약과 태그만 추출)
        summary_match = re.search(r"요약:\n(.*?)\n\n", content, re.DOTALL)
        tag_match = re.search(r"추천 태그:\n(.*?)\n\n", content, re.DOTALL)
        
        bert_summary['summary'] = summary_match.group(1).strip() if summary_match else "BERT 요약 결과를 찾을 수 없습니다."
        bert_summary['tags'] = [t.strip() for t in tag_match.group(1).split(',') if tag_match] if tag_match else []

    except FileNotFoundError:
        return df, None, f"⚠️ BERT 분석 파일을 찾을 수 없습니다: {txt_path}"
    
    return df, bert_summary, None

# 개인화 로직: 사용자 선호도 기반 리뷰 점수 계산
def get_personalized_recommendation(df, user_persona_vector):
    """
    사용자 성향 벡터와 각 리뷰의 성향 벡터를 비교하여 점수화하고 개인화된 정보를 추출합니다.
    """
    # 1. 사용자 벡터 정규화 (총합 1)
    user_vector = np.array(list(user_persona_vector.values()))
    if np.sum(user_vector) > 0:
        user_vector = user_vector / np.sum(user_vector)
    
    # 2. 리뷰별 성향 벡터 추출
    review_vectors = df[[f'S_{axis}' for axis in PERSONA_AXES]].values
    
    # 3. 개인화 점수 계산 (Dot Product)
    # 사용자 선호도와 리뷰의 성향 일치 정도를 점수화
    df['personalized_score'] = np.dot(review_vectors, user_vector)
    
    # 4. 가장 선호도가 높은 상위 10개 리뷰 추출
    top_n = 10
    personalized_df = df.sort_values(by='personalized_score', ascending=False).head(top_n)
    
    # 5. 개인화된 요약 생성 (선호도 높은 리뷰의 키워드 기반)
    top_reviews_text = " ".join(personalized_df['review_text'].tolist())
    
    # 6. 개인화 태그 추출 (선호도가 높은 리뷰에서 자주 언급된 키워드 기반)
    # 이 부분은 BERT의 키워드 추출 기능을 대체하는 단순 키워드 카운팅으로 대체합니다.
    
    all_keywords = []
    # 예시로 선호 성향과 관련된 태그만 추출
    for axis in PERSONA_AXES:
        if user_persona_vector[axis] > 0.3: # 사용자가 강하게 선호하는 성향
            # generator_bert.py에 정의된 태그 후보를 임시로 사용
            TAG_CANDIDATES = {
                'narrative': ["#갓서사", "#스토리_몰입", "#감동적"],
                'freedom': ["#높은_자유도", "#탐험", "#나만의_선택"],
                'stability': ["#갓적화", "#버그없음", "#쾌적함"],
                'challenge': ["#핵심_난이도", "#도전의식", "#피지컬_게임"]
            }
            all_keywords.extend(TAG_CANDIDATES.get(axis, []))

    return top_reviews_text, list(set(all_keywords)) # 중복 제거

# Streamlit 앱 레이아웃
def app():
    st.set_page_config(page_title="게임 리뷰 성향 분석기", layout="wide")
    st.title("🎮 게임 리뷰 성향 기반 분석 및 추천기")
    st.markdown("---")

    # 1. 게임 파일 선택
    try:
        # dataSet 디렉토리에서 'analyzed_'로 시작하는 CSV 파일 목록 로드
        available_files = [f.replace('analyzed_', '').replace('_reviews.csv', '') 
                           for f in os.listdir(DATA_DIR) if f.startswith('analyzed_') and f.endswith('.csv')]
    except FileNotFoundError:
        available_files = []
        st.error(f"❌ '{DATA_DIR}' 디렉토리를 찾을 수 없습니다. 크롤링 및 분석을 먼저 진행해주세요.")
        return

    if not available_files:
        st.warning("분석된 게임 파일이 없습니다. `crawler.py`와 Jupyter Notebook 분석을 먼저 실행하세요.")
        return

    game_name_select = st.selectbox("분석할 게임을 선택하세요:", available_files)
    
    # 데이터 로드
    df, bert_summary, error_message = load_data(game_name_select)

    if error_message:
        st.error(error_message)
        return
    
    # --- 2. 사용자 성향 입력 ---
    st.header("👤 사용자 선호 성향 입력")
    st.markdown("각 축을 조절하여 **'사용자님'이 게임을 선택할 때 중요하게 생각하는 요소**의 비중을 설정해주세요. (총합은 무시됨)")

    user_persona_input = {}
    cols = st.columns(4)
    
    # 슬라이더를 통해 4가지 성향 입력
    for i, axis in enumerate(PERSONA_AXES):
        with cols[i]:
            user_persona_input[axis] = st.slider(
                label=PERSONA_LABELS_KO[axis],
                min_value=0,
                max_value=10,
                value=5,
                key=axis
            )
            
    # 버튼 클릭 시 분석 시작
    if st.button("✨ 개인화 분석 시작"):
        
        st.markdown("---")
        st.header(f"결과: '{game_name_select}' 맞춤형 분석")
        
        # --- 3. 전체 요약 (BERT 결과) ---
        st.subheader("📝 전체 리뷰 기반 게임 요약 (BERT)")
        if bert_summary and bert_summary.get('summary'):
            st.info(bert_summary['summary'])
        else:
            st.warning("BERT 요약 분석 결과가 로드되지 않았습니다.")
            
        # --- 4. 개인화된 추천 태그 및 요약 ---
        user_vector_dict = {axis: user_persona_input[axis] for axis in PERSONA_AXES}
        top_reviews_text, personalized_tags = get_personalized_recommendation(df, user_vector_dict)
        
        st.subheader("💡 사용자 맞춤형 추천 태그")
        
        # 선호 태그 출력
        tag_display = " ".join([f'<span style="background-color:#007BFF; color:white; padding:5px 10px; border-radius:15px; margin-right:5px; font-weight:bold;">{tag}</span>' for tag in personalized_tags])
        st.markdown(tag_display, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("📖 개인화 요약 및 추천 리뷰 (상위 10개 리뷰 기반)")
        st.write(f"**{game_name_select}** 게임은 사용자님이 선호하시는 **{', '.join([PERSONA_LABELS_KO[k] for k, v in user_vector_dict.items() if v >= 7])}** 성향의 리뷰어들로부터 높은 평가를 받았습니다. 사용자님과 취향이 비슷한 상위 10개 리뷰의 핵심 내용은 다음과 같습니다:")
        
        # 단순 요약 대신, BERT가 없으므로 상위 리뷰 텍스트를 보여줌
        st.code(top_reviews_text[:1000] + "...", language='text')

if __name__ == "__main__":
    app()