# generator_bert.py
# 실행방법: python generator_bert.py

import pandas as pd
import numpy as np
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

# --- 환경 설정 ---
# DATA_DIR = "dataSet"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataSet')

# 한국어 BERT 모델 (KoBERT) 로드
MODEL_NAME = "skt/kobert-base-v1" 

# Streamlit 환경에서 캐싱 가능하도록 별도 함수로 정의
def load_bert_model():
    """BERT 모델과 토크나이저를 로드합니다. (app.py에서 캐싱하여 사용)"""
    try:
        from kobert_transformers import get_tokenizer as get_kobert_tokenizer # 함수 내에서 다시 import
        tokenizer = get_kobert_tokenizer()
        model = AutoModel.from_pretrained("skt/kobert-base-v1")
        model.eval()
        return tokenizer, model
    except Exception as e:
        # Streamlit 환경이 아니므로 오류를 반환
        print(f"BERT 모델 로드 실패: {e}") 
        return None, None
    
# 최종 전처리 및 사용자별 성향 벡터 집계
def aggregate_user_profiles(df: pd.DataFrame, min_playtime_hours: int = 2) -> Tuple[Dict[str, float], List[str]]:
    """
    개별 리뷰 데이터를 사용자(author_id)별 최종 성향 벡터로 집계합니다.
    """
    # try:
    #     df = pd.read_csv(input_csv_path)
    # except FileNotFoundError:
    #     print(f"❌ 오류: 파일을 찾을 수 없습니다: {input_csv_path}")
    #     return None
    
    df['playtime_hours'] = df['playtime_forever'] / 60
    
    # 1) 신뢰도 낮은 데이터 필터링 (플레이 시간 2시간 미만(게임환불시간:2h), 텍스트 없음)
    df_filtered = df[
        (df['playtime_hours'] >= min_playtime_hours) & 
        (df['review_text'].str.strip() != '') &
        (df['review_text'].notna())
    ].copy()

    if df_filtered.empty:
        # print("경고: 필터링 후 분석할 유효한 리뷰가 부족합니다.")
        return {}, []
    
    # print(f"✅ 최종 분석 대상 리뷰 수: {len(df_filtered)}개")
    
    # 성향 벡터 컬럼 목록
    persona_cols = [col for col in df_filtered.columns if col.startswith('S_')]
    
    # 2) 사용자별 최종 성향 벡터 및 통계 집계
    agg_args = {
        'review_text': lambda x: x.loc[x.str.len().idxmax()] if x.str.len().max() > 0 else '',
        'voted_up': 'mean',
        'playtime_forever': 'mean'
    }
    for col in persona_cols:
        agg_args[col] = 'mean'
    
    agg_df = df_filtered.groupby('author_id').agg(agg_args).reset_index()
    
    # 전체 유저의 최종 평균 성향 벡터 (요약 및 태그 생성에 사용)
    game_persona_vector = agg_df[persona_cols].mean().to_dict()
    
    # # 게임 이름 추출
    # # 예: analyzed_산나비_reviews.csv
    # parts = os.path.basename(input_csv_path).split('_')
    # game_name = parts[-2]
    
    # 모든 리뷰 텍스트를 하나의 리스트로 반환 (요약에 사용)
    all_reviews = df['review_text'].fillna('').tolist()
    
    return game_persona_vector, all_reviews

# BERT 기반 추출 요약 (Extractive Summarization)
def get_sentence_embeddings(reviews: List[str], tokenizer: AutoTokenizer, model: AutoModel) -> Tuple[List[str], np.ndarray | None]:
    """문장을 BERT 모델을 통해 임베딩 벡터로 변환합니다."""
    # 모든 리뷰를 문장 단위로 분리
    sentences = []
    for review in reviews:
        if review is None or not isinstance(review, str):
            continue
        # 간단한 문장 분리 (마침표 기준)
        sentences.extend([s.strip() for s in review.split('.') if s.strip()])

    # 문장이 너무 많으면 메모리 문제 발생 가능, 최대 500개 문장만 사용
    sentences = sentences[:500] 
    
    if not sentences:
        return [], None
    
    # 토큰화 및 임베딩 생성
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] 토큰의 출력을 문장 임베딩으로 사용
        embeddings = outputs.last_hidden_state[:, 0, :].numpy() 
        
    return sentences, embeddings

def generate_summary(sentences: List[str], embeddings: np.ndarray | None, summary_length: int = 4) -> str:
    """K-Means 클러스터링을 사용하여 가장 대표적인 문장을 추출합니다."""
    
    if not sentences or len(sentences) < summary_length:
        return "리뷰 텍스트가 부족하거나 너무 짧아 요약 생성에 실패했습니다."
    try:
        # K-Means 클러스터링 (요약 문장 수 = 클러스터 수)
        kmeans = KMeans(n_clusters=summary_length, random_state=42, n_init='auto')
        kmeans.fit(embeddings)
        
        # 각 클러스터의 중심(Centroid)을 찾습니다.
        centroids = kmeans.cluster_centers_
        
        # 각 클러스터 중심과 가장 가까운 문장을 찾습니다. (대표 문장)
        summary_sentences = []
        
        for i in range(summary_length):
            # 해당 클러스터에 속하는 임베딩 벡터와 중심 벡터 간의 거리 계산
            distances = cosine_similarity([centroids[i]], embeddings)[0]
            
            # 거리가 가장 가까운 문장의 인덱스 (가장 유사한 문장)
            closest_index = np.argsort(distances)[-1] 
            
            # 중복 방지
            if sentences[closest_index] not in summary_sentences:
                summary_sentences.append(sentences[closest_index])
                
        return ". ".join(summary_sentences) + "."
    except Exception as e:
            return f"요약 중 오류 발생: {e}"

# 성향 벡터 기반 태그 예측
def predict_tags(persona_vector: Dict[str, float]) -> List[str]:
    """성향 벡터를 기반으로 추천 태그를 생성합니다."""
    
    # 임계값
    THRESHOLD = 0.15 
    
    # 미리 정의된 태그 후보 목록
    TAG_CANDIDATES = {
        'narrative': ["#갓서사", "#스토리_중심", "#뛰어난_연출"],
        'freedom': ["#높은_자유도", "#탐험_필수", "#나만의_선택", "#오픈월드"],
        'stability': ["#갓적화", "#쾌적한_환경", "#버그_없음", "#안정적"],
        'challenge': ["#극악의_난이도", "#도전과제", "#피지컬_요구", "#고수전용"]
    }
    
    tag_scores = {}
    # 1. 1차 시도: 임계값 0.15 이상인 태그만 수집
    for vector_key, score in persona_vector.items():
        key_without_s = vector_key.replace('S_', '')
        if key_without_s in TAG_CANDIDATES and score >= THRESHOLD:
            for tag in TAG_CANDIDATES[key_without_s]:
                tag_scores[tag] = tag_scores.get(tag, 0) + score
    
    # 2. 2차 시도: 임계값 0.15를 넘지 못했다면, 최고점을 가진 축 1개만 선택 (폴백)
    if not tag_scores:
        # 벡터 값이 가장 높은 축과 그 값 찾기
        best_axis = max(persona_vector, key=persona_vector.get)
        best_score = persona_vector[best_axis]
            
        # 최고점 축의 첫 번째 태그만 선택
        key_without_s = best_axis.replace('S_', '')
        if key_without_s in TAG_CANDIDATES:
            # 해당 축의 대표 태그 1개만 반환
            return [TAG_CANDIDATES[key_without_s][0]]
                    
    # 점수가 높은 상위 5개 태그 선택
    sorted_tags = sorted(tag_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    
    # 태그 목록만 반환
    return [tag for tag, score in sorted_tags]

def run_bert_generation(analyzed_csv_path: str, game_name: str, tokenizer: AutoTokenizer, model: AutoModel) -> Tuple[str, List[str], str, float]:
    """
    분석된 CSV 파일을 로드하여 BERT 생성 및 TXT 파일 저장을 실행합니다.
    """
    try:
        df = pd.read_csv(analyzed_csv_path)
    except Exception as e:
        raise FileNotFoundError(f"분석 파일 로드 실패: {analyzed_csv_path}. 오류: {e}")
    
    if df.empty:
        raise ValueError("로드된 분석 데이터프레임이 비어있습니다. 리뷰 수집을 확인하세요.")
    
    # 1.1. 전체 긍정 비율 계산 (모든 리뷰 사용)
    total_reviews = len(df)
    # 'voted_up' (True/False)를 1/0으로 변환하여 합산
    df['voted_up'] = df['voted_up'].astype(int)
    positive_count = df['voted_up'].sum() 
    pos_ratio = round((positive_count / total_reviews) * 100, 1) if total_reviews > 0 else 0.0

    # 1.2. 최종 전처리 및 사용자별 성향 벡터 집계
    game_persona_vector, all_reviews = aggregate_user_profiles(df)
    
    if not game_persona_vector:
        # 유효한 리뷰가 없을 경우 처리
        safe_game_name = game_name.replace(' ', '_')
        output_filename = os.path.join(DATA_DIR, f"BERT_Analysis_{safe_game_name}.txt")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"게임: {game_name}\n")
            f.write(f"긍정 비율: {pos_ratio}%\n\n")  # 💡 긍정 비율 추가
            f.write("요약:\n유효한 리뷰(플레이 시간 5시간 이상)가 부족하여 분석을 건너뜁니다.\n\n")
            f.write("추천 태그:\n\n성향 벡터:\n{}")
            
        return "분석할 유효한 리뷰가 부족하여 요약 생성에 실패했습니다.", [], ""
    
    # 2. 요약 생성 (BERT)
    sentences, embeddings = get_sentence_embeddings(all_reviews, tokenizer, model)
    summary = generate_summary(sentences, embeddings, summary_length=4)
    
    # 3. 태그 생성 (성향 벡터 기반)
    predicted_tags = predict_tags(game_persona_vector)
    
    # 4. 결과 저장
    safe_game_name = game_name.replace(' ', '_')
    output_filename = os.path.join(DATA_DIR, f"BERT_Analysis_{safe_game_name}.txt")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"게임: {game_name}\n")
        f.write(f"긍정 비율: {pos_ratio}%\n\n")
        f.write(f"요약:\n{summary}\n\n")
        f.write(f"추천 태그:\n{', '.join(predicted_tags)}\n\n")
        f.write(f"성향 벡터:\n{json.dumps(game_persona_vector, ensure_ascii=False, indent=4)}\n")
    
    return summary, predicted_tags, output_filename, pos_ratio

# def main_generator_bert():
#     # 파일 경로 설정 (dataSet 디렉토리의 파일을 로드)
#     input_filename = input("분석된 CSV 파일명을 입력하세요 (예: analyzed_{appID}_reviews.csv): ")
#     input_csv_path = os.path.join(DATA_DIR, input_filename)

#     # 1. 최종 전처리 및 성향 벡터 집계
#     result = aggregate_user_profiles(input_csv_path)
    
#     if result is None:
#         print("분석을 위한 유효한 사용자 프로필을 생성하지 못했습니다.")
#         return

#     game_name, persona_vector, all_reviews = result
    
#     # 2. BERT 모델 로드
#     print(f"\n! {MODEL_NAME} 모델 로드 중...")
#     try:
#         tokenizer = get_kobert_tokenizer()
#         model = AutoModel.from_pretrained(MODEL_NAME)
#         model.eval() # 추론 모드
#     except Exception as e:
#         print(f"❌ BERT 모델 로드 실패: {e}")
#         return

#     # 3. 요약 생성 (BERT 기반 추출 요약)
#     sentences, embeddings = get_sentence_embeddings(all_reviews, tokenizer, model)
#     summary = generate_summary(sentences, embeddings, summary_length=4)
    
#     # 4. 태그 생성 (성향 벡터 기반 예측)
#     predicted_tags = predict_tags(persona_vector)
    
#     # 5. 결과 출력 및 저장
#     print("\n========================================")
#     print(f"🎮 게임 분석 결과: {game_name}")
#     print("========================================")
#     print("1. 추출 요약:")
#     print(summary)
#     print("\n2. 성향 벡터 기반 추천 태그:")
#     print(f"{', '.join(predicted_tags)}")
#     print("\n3. 최종 성향 벡터:")
#     print(json.dumps({k.replace('S_', ''): round(v, 4) for k, v in persona_vector.items()}, indent=4, ensure_ascii=False))
#     print("========================================")
    
#     # 결과를 텍스트 파일로 저장
#     output_filename = os.path.join(DATA_DIR, f"BERT_Analysis_{game_name}.txt")
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         f.write(f"게임: {game_name}\n")
#         f.write(f"요약:\n{summary}\n\n")
#         f.write(f"추천 태그:\n{', '.join(predicted_tags)}\n\n")
#         f.write(f"성향 벡터:\n{json.dumps(persona_vector, ensure_ascii=False, indent=4)}\n")
    
#     print(f"\n📂 분석 결과가 '{output_filename}'에 저장되었습니다.")


# if __name__ == "__main__":
#     main_generator_bert()