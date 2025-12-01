# collector.py

import requests
from bs4 import BeautifulSoup
import steamreviews
import os
import json
import pandas as pd
from typing import List, Dict, Any

# --- 환경 설정 ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataSet')

# --- 크롤링 함수 ---
def get_app_id_by_name(game_name: str) -> int | None:
    """Steam 상점 검색 페이지를 스크래핑하여 App ID를 찾습니다."""
    search_url = f"https://store.steampowered.com/search/?term={game_name}&supportedlang=koreana"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        first_result = soup.find('a', class_='search_result_row')
        
        if first_result:
            app_id = first_result.get('data-ds-appid')
            return int(app_id) if app_id else None
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

def get_game_reviews(app_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """특정 게임(app_id)의 한국어 리뷰를 수집합니다."""
    request_params = dict(
        language='koreana', 
        filter='recent',
        num_per_page=100
    )
    
    review_dict, _ = steamreviews.download_reviews_for_app_id(
        app_id, 
        chosen_request_params=request_params,
        verbose=False
    )
    
    reviews_data = []
    if 'reviews' in review_dict:
        sorted_reviews = sorted(review_dict['reviews'].items(), key=lambda x: x[0], reverse=True) 
        
        count = 0
        for review_id, review in sorted_reviews:
            if count >= limit: break
            
            reviews_data.append({
                'review_id': review_id,
                'author_id': review['author']['steamid'],
                'playtime_forever': review['author']['playtime_forever'],
                'review_text': review['review'],
                'voted_up': review['voted_up']
            })
            count += 1
            
    return reviews_data

def run_collection(game_name: str, limit: int = 200) -> tuple[str | None, int | None, str | None]:
    """크롤링을 실행하고 원본 JSON 파일 경로와 App ID를 반환합니다."""
    app_id = get_app_id_by_name(game_name)
    if not app_id:
        return None, None, f"Steam에서 '{game_name}'의 App ID를 찾을 수 없습니다."

    reviews_data = get_game_reviews(app_id, limit=limit)
    if not reviews_data:
        return None, None, "리뷰 데이터를 수집하지 못했습니다."

    safe_game_name = game_name.replace(' ', '_')
    output_filename = f"reviews_{app_id}_{limit}_{safe_game_name}.json"
    output_path = os.path.join(DATA_DIR, output_filename)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reviews_data, f, ensure_ascii=False, indent=4)
        
    return output_path, app_id, None