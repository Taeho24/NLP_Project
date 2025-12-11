# collector.py

import requests
from bs4 import BeautifulSoup
import steamreviews
import os
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import time

# --- 환경 설정 ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataSet')
MAX_RETRIES = 3 # 최대 재시도 횟수

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

def get_game_reviews(app_id: int, limit: int = 200) -> Tuple[Dict[str, Any] | None, str | None]:
    """특정 게임(app_id)의 한국어 리뷰를 수집합니다."""
    request_params = dict(
        language='korean', 
        filter='all',
        num_per_page=100
    )
    for attempt in range(MAX_RETRIES):
        try:
            review_dict, query_count = steamreviews.download_reviews_for_app_id(
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
                        'author_id': review.get('author', {}).get('steamid'),
                        'playtime_forever': review.get('author', {}).get('playtime_forever', 0),
                        'review_text': review.get('review', ''), 
                        'voted_up': review.get('voted_up', False)
                    })
                    count += 1
                    
            if reviews_data:
                # 리뷰 딕셔너리를 포함하여 run_collection으로 전달
                return {'reviews': reviews_data}, None 
            else:
                return None, f"리뷰를 찾을 수 없거나 수집 제한({limit}개)으로 인해 데이터가 비어있습니다."

        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt * 5 
                print(f"ConnectionError 발생. {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                return None, f"Steam API 연결 오류 (최대 재시도 {MAX_RETRIES}회 실패): {e}"
        
        except Exception as e:
            return None, f"리뷰 다운로드 중 예상치 못한 오류 발생: {e}"

    return None, "리뷰 다운로드 중 알 수 없는 오류 발생 (최종 실패)"

def search_games(game_name: str) -> List[Dict[str, Any]]:
    """
    Steam API를 사용하여 입력된 게임 이름과 연관된 최대 10개의 게임 목록을 반환합니다.
    """
    if not game_name:
        return []
    
    # Steam store API의 search endpoint 사용
    search_url = "https://store.steampowered.com/api/storesearch"
    params = {
        'cc': 'kr',           # 국가 코드 (한국)
        'l': 'korean',        # 언어 (한국어)
        'term': game_name,    # 검색어
        'request': '1',
        'f': 'json'
    }
    
    results = []
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status() # HTTP 오류 시 예외 발생
            data = response.json()
            
            if data and 'items' in data:
                # 이름, App ID, 이미지 URL, 가격 정보만 추출
                results = []
                for item in data['items'][:10]: # 최대 10개만 반환
                    app_id = item.get('id')
                    header_image_url = f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg"

                    results.append({
                        'app_id': app_id,
                        'name': item.get('name'),
                        'header_image': header_image_url,
                    })
                return results
            
            return []

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt * 5 
                print(f"Steam API 검색 중 ConnectionError 발생. {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                print(f"Steam API 검색 오류 (최대 재시도 {MAX_RETRIES}회 실패): {e}")
                return []
        
        except Exception as e:
            print(f"Steam API 검색 중 예상치 못한 오류 발생: {e}")
            return []

def run_collection(app_id: int, game_name: str, limit: int = 200) -> Tuple[str, int, str]:
    """크롤링을 실행하고 원본 JSON 파일 경로와 App ID를 반환합니다."""
    app_id = get_app_id_by_name(game_name)
    if not app_id:
        return None, None, f"Steam에서 '{game_name}'의 App ID를 찾을 수 없습니다."

    reviews_data_dict, error = get_game_reviews(app_id, limit=limit)
    
    if error: 
        print(f"리뷰 수집 오류 발생: {error}") # 💡 디버깅 로그 추가
        return None, None, f"리뷰 데이터 수집 오류: {error}"
    
    if not reviews_data_dict or 'reviews' not in reviews_data_dict:
        print(f"리뷰 데이터 딕셔너리가 비어있거나 'reviews' 키가 없습니다. App ID: {app_id}") # 💡 디버깅 로그 추가
        return None, None, "리뷰 데이터를 수집하지 못했거나 데이터 구조에 문제가 있습니다."
    
    safe_game_name = game_name.replace(' ', '_')
    output_filename = f"reviews_{app_id}_{limit}_{safe_game_name}.json"
    output_path = os.path.join(DATA_DIR, output_filename)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reviews_data_dict, f, ensure_ascii=False, indent=4)
        
    return output_path, app_id, None