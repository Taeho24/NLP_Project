# app.py
# 실행방법: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import sys
import torch
import json
from transformers import AutoModel
from kobert_transformers import get_tokenizer as get_kobert_tokenizer
from typing import Dict, List, Tuple, Any

# --- 모듈 경로 설정 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'collectData'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))

try:
    from collector import run_collection, search_games
    from analyzer import run_analysis
    from generator_bert import run_bert_generation, load_bert_model
except ImportError as e:
    st.error(f"❌ 모듈 로드 실패. collectData/ 또는 model/ 디렉토리 구조와 파일명을 확인하세요: {e}")
    sys.exit()

# --- 환경 설정 및 상수 ---
DATA_DIR = "dataSet"
PERSONA_AXES = ['narrative', 'freedom', 'stability', 'challenge']
PERSONA_LABELS_KO = {
    'narrative': '스토리/서사 선호',
    'freedom': '자유도/탐험 선호',
    'stability': '최적화/안정성 선호',
    'challenge': '도전/난이도 선호'
}

@st.cache_resource
def load_bert_resources():
    """BERT 모델과 토크나이저를 로드하고 캐싱합니다."""
    return load_bert_model()

def get_personalized_recommendation(df: pd.DataFrame, user_persona_vector: Dict[str, int]) -> Tuple[str, List[str], int, int]:
    """
    사용자 성향 벡터와 각 리뷰의 성향 벡터를 비교하여 개인화된 추천 태그 및 요약을 생성합니다.
    """
    user_vector = np.array(list(user_persona_vector.values()))
    if np.sum(user_vector) > 0:
        user_vector = user_vector / np.sum(user_vector)
    
    # 성향 벡터 칼럼명은 'S_narrative', 'S_freedom' 형태임
    review_vectors = df[[f'S_{axis}' for axis in PERSONA_AXES]].values
    
    # 개인화 점수 계산 (Dot Product)
    df['personalized_score'] = np.dot(review_vectors, user_vector)
    
    top_n = 10
    personalized_df = df.sort_values(by='personalized_score', ascending=False).head(top_n)
    
    # 상위 리뷰 텍스트 결합 (개인화 요약)
    top_reviews_text = " ".join(personalized_df['review_text'].tolist())
    
    # 개인화된 리뷰 10개의 긍정/부정 개수 계산
    pos_count = personalized_df['voted_up'].sum()
    neg_count = top_n - pos_count
    
    # 개인화 태그 추출 (사용자가 강하게 선호하는 성향 기반)
    all_keywords = []
    # 슬라이더 10점 만점 중 7점 이상을 강한 선호도로 간주
    for axis in PERSONA_AXES:
        if user_persona_vector[axis] >= 7: 
            TAG_CANDIDATES = {
                'narrative': ["#갓서사", "#스토리_몰입", "#감동적"],
                'freedom': ["#높은_자유도", "#탐험", "#나만의_선택"],
                'stability': ["#갓적화", "#버그없음", "#쾌적함"],
                'challenge': ["#핵심_난이도", "#도전의식", "#피지컬_게임"]
            }
            all_keywords.extend(TAG_CANDIDATES.get(axis, []))

    return top_reviews_text, list(set(all_keywords)), pos_count, neg_count

def load_summary_data(summary_txt_path: str) -> Dict[str, Any]:
    """분석 TXT 파일을 읽어 필요한 데이터를 추출합니다."""
    
    # 기본값 설정 (파싱 실패 시 이 메시지가 출력됩니다)
    summary_data = {
        'positive_ratio': None,
        'summary': 'BERT 분석 결과를 찾을 수 없습니다.',
        'tags': [],
        'persona_vector': {}
    }

    try:
        with open(summary_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 긍정 비율 추출
        ratio_match = re.search(r"긍정 비율: (\d+\.?\d*)%", content)
        if ratio_match:
            summary_data['positive_ratio'] = float(ratio_match.group(1)) / 100.0

        # 2. 요약 텍스트 추출 (요약:과 \n\n추천 태그: 사이)
        # re.DOTALL을 사용하여 줄바꿈 포함 모든 문자 매칭
        summary_match = re.search(r"요약:\s*\n(.*?)\n\n추천 태그:", content, re.DOTALL)
        if summary_match:
            summary_data['summary'] = summary_match.group(1).strip()
            
        # 3. 추천 태그 추출 (추천 태그:와 \n\n성향 벡터: 사이)
        tag_match = re.search(r"추천 태그:\s*\n(.*?)\n\n성향 벡터:", content, re.DOTALL)
        if tag_match:
            raw_tags = tag_match.group(1).strip()
            # 쉼표를 기준으로 분리 후, 각 태그에서 공백 제거
            summary_data['tags'] = [t.strip() for t in raw_tags.split(',') if t.strip()]
        
        # 4. 성향 벡터 추출 (성향 벡터: 다음에 오는 JSON 블록)
        # 성향 벡터: 다음에 오는 모든 내용을 JSON 문자열로 간주
        vector_match = re.search(r"성향 벡터:\s*\n(.*?)\s*$", content, re.DOTALL)
        if vector_match:
            try:
                json_string = vector_match.group(1).strip()
                summary_data['persona_vector'] = json.loads(json_string)
            except json.JSONDecodeError as e:
                # JSON 파싱 실패 시 디버깅 메시지 출력
                print(f"성향 벡터 JSON 파싱 오류: {e}") 
                summary_data['persona_vector'] = {}

    except FileNotFoundError:
        summary_data['summary'] = f"분석 결과 파일 ({summary_txt_path})을 찾을 수 없습니다."
    except Exception as e:
        summary_data['summary'] = f"분석 요약 파일 로드/파싱 중 예상치 못한 오류 발생: {e}"
        print(f"분석 요약 파일 로드/파싱 오류: {e}")

    return summary_data

# # 데이터 로드 및 캐싱
# @st.cache_data
# def load_data(filename):
#     """분석된 CSV 파일과 BERT 분석 텍스트 파일을 로드합니다."""
    
#     # 1-1. 분석된 리뷰 데이터 (CSV) 로드
#     csv_path = os.path.join(DATA_DIR, f"analyzed_{filename}_reviews.csv")
#     try:
#         df = pd.read_csv(csv_path)
#     except FileNotFoundError:
#         return None, None, f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}"

#     # 1-2. BERT 분석 결과 (TXT) 로드
#     txt_path = os.path.join(DATA_DIR, f"BERT_Analysis_{filename}.txt")
#     bert_summary = {}
#     try:
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             content = f.read()
            
#         # 텍스트 파일 내용 파싱 (간단하게 요약과 태그만 추출)
#         summary_match = re.search(r"요약:\n(.*?)\n\n", content, re.DOTALL)
#         tag_match = re.search(r"추천 태그:\n(.*?)\n\n", content, re.DOTALL)
        
#         bert_summary['summary'] = summary_match.group(1).strip() if summary_match else "BERT 요약 결과를 찾을 수 없습니다."
#         bert_summary['tags'] = [t.strip() for t in tag_match.group(1).split(',') if tag_match] if tag_match else []

#     except FileNotFoundError:
#         return df, None, f"⚠️ BERT 분석 파일을 찾을 수 없습니다: {txt_path}"
    
#     return df, bert_summary, None

# # 개인화 로직: 사용자 선호도 기반 리뷰 점수 계산
# def get_personalized_recommendation(df, user_persona_vector):
#     """
#     사용자 성향 벡터와 각 리뷰의 성향 벡터를 비교하여 점수화하고 개인화된 정보를 추출합니다.
#     """
#     # 1. 사용자 벡터 정규화 (총합 1)
#     user_vector = np.array(list(user_persona_vector.values()))
#     if np.sum(user_vector) > 0:
#         user_vector = user_vector / np.sum(user_vector)
    
#     # 2. 리뷰별 성향 벡터 추출
#     review_vectors = df[[f'S_{axis}' for axis in PERSONA_AXES]].values
    
#     # 3. 개인화 점수 계산 (Dot Product)
#     # 사용자 선호도와 리뷰의 성향 일치 정도를 점수화
#     df['personalized_score'] = np.dot(review_vectors, user_vector)
    
#     # 4. 가장 선호도가 높은 상위 10개 리뷰 추출
#     top_n = 10
#     personalized_df = df.sort_values(by='personalized_score', ascending=False).head(top_n)
    
#     # 5. 개인화된 요약 생성 (선호도 높은 리뷰의 키워드 기반)
#     top_reviews_text = " ".join(personalized_df['review_text'].tolist())
    
#     # 6. 개인화 태그 추출 (선호도가 높은 리뷰에서 자주 언급된 키워드 기반)
#     # 이 부분은 BERT의 키워드 추출 기능을 대체하는 단순 키워드 카운팅으로 대체합니다.
    
#     all_keywords = []
#     # 예시로 선호 성향과 관련된 태그만 추출
#     for axis in PERSONA_AXES:
#         if user_persona_vector[axis] > 0.3: # 사용자가 강하게 선호하는 성향
#             # generator_bert.py에 정의된 태그 후보를 임시로 사용
#             TAG_CANDIDATES = {
#                 'narrative': ["#갓서사", "#스토리_몰입", "#감동적"],
#                 'freedom': ["#높은_자유도", "#탐험", "#나만의_선택"],
#                 'stability': ["#갓적화", "#버그없음", "#쾌적함"],
#                 'challenge': ["#핵심_난이도", "#도전의식", "#피지컬_게임"]
#             }
#             all_keywords.extend(TAG_CANDIDATES.get(axis, []))

#     return top_reviews_text, list(set(all_keywords)) # 중복 제거

# Streamlit 앱 레이아웃
def main_app():
    st.set_page_config(page_title="게임 리뷰 성향 분석기", layout="wide")
    st.title("🎮 게임 리뷰 통합 분석 파이프라인")
    st.markdown("---")

    # BERT 모델 사전 로드 및 캐싱
    tokenizer, model = load_bert_resources()
    if tokenizer is None or model is None:
        st.warning("BERT 모델 로드에 실패했습니다. 파이프라인 실행이 불가능합니다.")
        st.stop() 

    # 1. 신규 게임 크롤링 및 분석
    st.header("1. 신규 게임 크롤링 및 분석")
    search_term = st.text_input("분석할 **게임 이름**을 입력하고 [검색] 버튼을 누르세요:", key="search_term")
    review_limit = st.slider("수집할 리뷰 개수 (최대 500개)", 50, 500, 200, step=50)
    
    if st.button("🔍 게임 검색"):
        if not search_term:
            st.warning("검색할 게임 이름을 입력해주세요.")
            st.session_state['search_results'] = []
            return
            
        with st.spinner(f"'{search_term}' 검색 중..."):
            results = search_games(search_term) # 💡 새로운 검색 함수 호출
            
        if not results:
            st.error(f"❌ '{search_term}'에 대한 검색 결과를 찾을 수 없습니다.")
            st.session_state['search_results'] = []
        else:
            st.session_state['search_results'] = results
            st.success(f"✅ 총 {len(results)}개의 연관 게임을 찾았습니다. 목록에서 게임을 선택하세요.")
            
    if 'search_results' in st.session_state and st.session_state['search_results']:
        st.subheader("검색 결과 및 분석할 게임 선택")
        # Streamlit의 columns를 사용하여 결과 표시
        
        # 선택된 게임 정보를 저장할 변수
        selected_game_info = None 
        
        # 각 검색 결과를 카드 형태로 표시
        for i, item in enumerate(st.session_state['search_results']):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # 이미지가 존재하는지 확인 (None이 아니고 빈 문자열이 아닐 때)
                if item.get('header_image'):
                    st.image(item['header_image'], width=100, caption=str(item['app_id']))
                else:
                    # 이미지가 없을 경우 대체 텍스트나 빈 공간 표시
                    st.write("🖼️ (No Image)")
            
            with col2:
                st.markdown(f"{item['name']}")
                st.markdown(
                    f"""
                    - **App ID:** {item['app_id']}
                    """
                )
                # 분석 시작 버튼 추가 (각 카드별로 버튼이 생김)
                if st.button(f"🚀 {item['name']} 분석 시작", key=f"analyze_btn_{item['app_id']}"):
                    selected_game_info = item
                    break # 버튼이 눌리면 루프 종료
                
        if selected_game_info:
            new_game_name = selected_game_info['name']
            app_id = selected_game_info['app_id']
            
            with st.spinner(f"게임 '{new_game_name}' 분석 파이프라인 실행 중 (ID: {app_id})..."):                
                # a. 크롤링 (run_collection 호출 - 이제 App ID를 직접 전달)
                json_path, app_id, error = run_collection(app_id, new_game_name, limit=review_limit)
                if error: st.error(f"❌ 크롤링 오류: {error}"); return
                st.success(f"✅ App ID 발견: {app_id}")
                
                # b. 성향 분석 (analyzer.py 호출)
                analyzed_path, error = run_analysis(json_path, app_id, new_game_name)
                if error: st.error(f"❌ 분석 오류: {error}"); return
                st.success(f"✅ 성향 벡터 분석 완료")
                
                # c. BERT 생성 (generator_bert.py 호출)
                summary, tags, output_path, pos_ratio = run_bert_generation(analyzed_path, new_game_name, tokenizer, model)

                st.success(f"✅ 파이프라인 분석 완료! 긍정 비율: {pos_ratio}%, 요약: {summary[:50]}...")
                st.balloons()
                st.session_state['last_analyzed_game'] = new_game_name
                st.session_state['search_results'] = [] # 분석 완료 후 검색 결과 초기화
                st.rerun()
                
    st.markdown("---")
    
    # 2. 분석된 게임 선택 및 개인화 분석 섹션
    st.header("2. 분석된 게임 선택 및 개인화")

    try:
        analyzed_files = [f for f in os.listdir(DATA_DIR) if f.startswith('analyzed_') and f.endswith('.csv')]
        game_options = {}
        for f in analyzed_files:
            base_name = f.replace('.csv', '').replace('_reviews', '')
            parts = base_name.split('_', 2)
            
            if len(parts) == 3:
                game_name_with_underscores = parts[2]
                game_name = game_name_with_underscores.replace('_', ' ') # 공백으로 변환
            else:
                # 파일 이름 형식이 예상과 다를 경우 폴백
                game_name = f"Unknown Game ({f})" 
            
            game_options[game_name] = f
            
        available_games = list(game_options.keys())

    except FileNotFoundError:
        available_games = []

    if not available_games:
        st.warning("분석된 게임 파일이 없습니다. 1번에서 새 게임을 분석하세요.")
        return
        
    last_game_name = st.session_state.get('last_analyzed_game')

    if last_game_name and last_game_name in available_games:
        default_index = available_games.index(last_game_name)
    else:
        # 💡 available_games 리스트가 비어있지 않은지 확인 후 0을 기본값으로 사용
        default_index = 0
        
    game_name_select = st.selectbox("개인화 분석할 게임을 선택하세요:", available_games, index=default_index)
    
    # 데이터 로드
    selected_csv_file = game_options[game_name_select]
    df = pd.read_csv(os.path.join(DATA_DIR, selected_csv_file))
    
    base_csv_name_no_suffix = selected_csv_file.replace('.csv', '').replace('_reviews', '')
    
    try:
        safe_game_name_with_underscore = base_csv_name_no_suffix.split('_', 2)[2] 
    except IndexError:
        st.error("분석 파일 이름 파싱에 실패했습니다. 파일명 형식을 확인하세요.")
        return

    # BERT 분석 결과 로드 (TXT 파일)
    txt_filename = f"BERT_Analysis_{safe_game_name_with_underscore}.txt"
    txt_path = os.path.join(DATA_DIR, txt_filename)

    summary_data = load_summary_data(txt_path) 

    bert_pos_ratio = summary_data['positive_ratio'] * 100.0 if summary_data['positive_ratio'] is not None else None
    bert_summary = summary_data['summary']
    bert_tags = summary_data['tags']
    # bert_persona_vector = summary_data['persona_vector']

    # --- 3. 사용자 성향 입력 ---
    st.header(f"👤 사용자 맞춤형 분석 ({game_name_select})")
    
    user_persona_input = {}
    cols = st.columns(4)
    
    for i, axis in enumerate(PERSONA_AXES):
        with cols[i]:
            user_persona_input[axis] = st.slider(
                label=PERSONA_LABELS_KO[axis],
                min_value=0,
                max_value=10,
                value=5,
                key=f"user_slider_{axis}"
            )
            
    if st.button(f"✨ '{game_name_select}' 개인화 분석 실행"):
        
        st.markdown("---")
        
        # 전체 리뷰 기반 긍정/부정 지표 표시
        st.subheader("👍 게임 전반적인 긍정/부정 지표")
        if bert_pos_ratio is not None:
            st.metric(
                label=f"전체 긍정 비율 ({len(df)} 리뷰 기준)", 
                value=f"{bert_pos_ratio}%",
                delta="좋아요! 이 게임은 평균적으로 긍정적인 평가를 받았습니다." if bert_pos_ratio >= 70 else "참고하세요. 리뷰가 긍정/부정으로 나뉘고 있습니다.",
            )
        else:
            st.warning("전체 긍정 비율을 로드할 수 없습니다.")
        
        st.markdown("---")
        
        # 전체 요약
        st.subheader("📝 전체 리뷰 기반 게임 요약 (BERT)")
        st.info(bert_summary)
        
        st.subheader("🔑 전체 리뷰 기반 추천 태그")
        tag_display_bert = " ".join([f'<span style="background-color:#5cb85c; color:white; padding:5px 10px; border-radius:15px; margin-right:5px; font-weight:bold;">{tag}</span>' for tag in bert_tags])
        if bert_tags: # 태그 리스트가 비어 있지 않으면 표시
            st.markdown(tag_display_bert, unsafe_allow_html=True)
        else: # 💡 태그 리스트가 비어 있다면 폴백 메시지 출력
            st.warning("분석 결과, 두드러지는 사용자 성향(임계값 0.15 이상)을 찾지 못하여 새로운 태그를 생성하지 않았습니다.")
        
        st.markdown("---")
        
        # 개인화 분석 실행
        user_vector_dict = {axis: user_persona_input[axis] for axis in PERSONA_AXES}
        top_reviews_text, personalized_tags, pos_count, neg_count = get_personalized_recommendation(df, user_vector_dict)
        
        st.subheader("💡 사용자 맞춤형 추천 태그")
        tag_display_personal = " ".join([f'<span style="background-color:#f0ad4e; color:white; padding:5px 10px; border-radius:15px; margin-right:5px; font-weight:bold;">{tag}</span>' for tag in personalized_tags])
        if personalized_tags:
            st.markdown(tag_display_personal, unsafe_allow_html=True)
        else:
            st.warning("사용자님과 강하게 일치하는 맞춤형 태그를 찾지 못했습니다. 선호도 슬라이더를 7점 이상으로 설정해 보세요.")
        
        st.subheader("📖 개인화 요약 및 추천 리뷰")
        
        st.markdown(f"**사용자님과 유사한 리뷰어 {pos_count + neg_count}명의 긍/부정 비율:**")
        col_pos, col_neg = st.columns(2)
        col_pos.metric("👍 긍정 리뷰", pos_count)
        col_neg.metric("👎 부정 리뷰", neg_count)
        st.markdown("---")
        
        strong_prefs = [PERSONA_LABELS_KO[k] for k, v in user_vector_dict.items() if v >= 7]
        
        if strong_prefs:
            st.write(f"사용자님과 취향(강력 선호: **{', '.join(strong_prefs)}**)이 비슷한 리뷰어들의 핵심 내용은 다음과 같습니다:")
        else:
            st.write("사용자님과 취향이 비슷한 상위 리뷰들의 핵심 내용은 다음과 같습니다:")

        st.code(top_reviews_text[:1200] + "..." if len(top_reviews_text) > 1200 else top_reviews_text, language='text')


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    main_app()