# analyzer.py

import pandas as pd
import numpy as np
import os
import re
import json
from typing import Dict, Any, Tuple

# --- 환경 설정 ---
KEYWORDS = {
    "narrative": ["스토리", "서사", "감동", "엔딩", "캐릭터", "몰입", "대사", "연출", "배경"],
    "freedom": ["자유도", "오픈월드", "탐험", "상호작용", "선택", "커스터마이징", "비선형"],
    "stability": ["최적화", "버그", "프레임", "렉", "튕김", "서버", "운영", "잔렉", "불안정"],
    "challenge": ["난이도", "컨트롤", "보스", "피지컬", "패턴", "도전", "소울", "어려움", "노력"]
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataSet') # Generator와 동일

# --- 성향 벡터 계산 함수 ---
def calculate_persona_vector(review_text: str) -> Dict[str, float]:
    """단일 리뷰 텍스트를 분석하여 성향 벡터 (4축 점수)를 반환합니다."""
    scores = {k: 0 for k in KEYWORDS.keys()}
    total_hits = 0
    
    if pd.isna(review_text) or not review_text.strip():
        return {k: 0.0 for k in KEYWORDS.keys()}

    for category, words in KEYWORDS.items():
        for word in words:
            count = len(re.findall(word, review_text, re.IGNORECASE))
            scores[category] += count
            total_hits += count
            
    if total_hits == 0:
        return {k: 0.0 for k in KEYWORDS.keys()}
    
    persona_vector = {k: round(v / total_hits, 3) for k, v in scores.items()}
    return persona_vector

def run_analysis(input_json_path: str, app_id: int, game_name: str) -> Tuple[str | None, str | None]:
    """
    JSON 파일을 로드하고 성향 벡터를 계산하여 CSV로 저장합니다.
    
    Args:
        json_path (str): 크롤링된 리뷰 JSON 파일 경로.
        app_id (int): Steam App ID.
        game_name (str): 게임 이름.

    Returns:
        Tuple[str | None, str | None]: (출력 CSV 경로 또는 None, 오류 메시지 또는 None)
    """
    # 1. JSON 파일 로드 및 유효성 검사
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
             
    except FileNotFoundError:
        return None, "리뷰 JSON 파일을 찾을 수 없습니다. (크롤링이 성공했는지 확인하세요.)"
    except json.JSONDecodeError:
        return None, "리뷰 JSON 파일이 비어있거나 JSON 형식 오류가 있습니다."
    except Exception as e:
        return None, f"리뷰 JSON 파일 로드 중 예상치 못한 오류: {e}"
    
    # 2. 리뷰 데이터 추출 및 유효성 검사
    reviews_data = []
    
    if isinstance(data, dict):
        # 💡 정상 경로: { 'reviews': [...] } 형태일 때
        reviews_data = data.get('reviews', [])
    elif isinstance(data, list):
        # 💡 방어 경로: 데이터가 리스트 자체로 저장되었을 때
        reviews_data = data
    else:
        # dict, list 모두 아닐 때 오류 처리
        return None, f"리뷰 파일의 최상위 데이터 형식이 올바르지 않습니다. (App ID: {app_id}, 타입: {type(data)})"
    
    if not reviews_data:
        return None, f"분석할 리뷰 데이터가 없습니다. (App ID: {app_id})"

    filtered_reviews_data = [
        item for item in reviews_data 
        if isinstance(item, dict) and item
    ]

    if not filtered_reviews_data:
        return None, f"필터링 후 유효한 리뷰 데이터가 없습니다. (App ID: {app_id})"
    
    # 3. DataFrame 생성
    try:
        df = pd.DataFrame(filtered_reviews_data)
    except Exception as e:
        return None, f"Pandas DataFrame 생성 오류: 수집된 데이터 형식이 다릅니다. 오류: {e}"
    
    # 4. 성향 벡터 계산 및 합치기
    if 'review_text' not in df.columns:
        print(f"🚨 KeyError 발생! DataFrame 컬럼: {df.columns.tolist()}")
        return None, f"분석 오류: 필수 컬럼 'review_text'를 찾을 수 없습니다. 실제 컬럼: {df.columns.tolist()}"
    df['persona_vector'] = df['review_text'].apply(calculate_persona_vector)
    
    # 5. 벡터 분해 및 DF 결합
    vector_df = df['persona_vector'].apply(pd.Series)
    vector_df.columns = ['S_' + col for col in vector_df.columns]
    df = pd.concat([df.drop(columns=['persona_vector']), vector_df], axis=1) # persona_vector 열 제거 후 결합
    
    # 6. CSV 파일 저장
    safe_game_name = game_name.replace(' ', '_')
    output_filename = f"analyzed_{app_id}_{safe_game_name}_reviews.csv"
    
    output_path = os.path.join(DATA_DIR, output_filename) 
    
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
    except Exception as e:
        return None, f"분석 결과 CSV 저장 오류: {e}"
    
    return output_path, None