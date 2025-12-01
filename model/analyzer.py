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
    """JSON 파일을 로드하고 성향 벡터를 계산하여 CSV로 저장합니다."""
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
    except FileNotFoundError:
        return None, "JSON 파일을 찾을 수 없습니다."
    
    # reviews_data = data.get('reviews', data) if isinstance(data, dict) else data
    df = pd.DataFrame(reviews_data)
    
    if df.empty:
        return None, "리뷰 데이터가 없습니다."

    df['persona_vector'] = df['review_text'].apply(calculate_persona_vector)
    vector_df = df['persona_vector'].apply(pd.Series)
    vector_df.columns = ['S_' + col for col in vector_df.columns]
    df = pd.concat([df, vector_df], axis=1)
    
    safe_game_name = game_name.replace(' ', '_')
    output_filename = f"analyzed_{app_id}_{safe_game_name}_reviews.csv"
    output_path = os.path.join(DATA_DIR, output_filename)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return output_path, None