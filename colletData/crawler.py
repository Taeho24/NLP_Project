# crawler.py
import steamreviews
import requests
import json
import sys
from bs4 import BeautifulSoup # HTML 파싱을 위해 새로 추가

# 필요한 라이브러리가 설치되어 있는지 확인
try:
    import requests
except ImportError:
    print("requests 라이브러리가 설치되어 있지 않습니다. pip install requests를 실행해주세요.")
    sys.exit(1)

# ==========================================
# 1단계: 게임 이름으로 App ID 검색

def get_app_id_by_name(game_name):
    """Steam 상점 검색 페이지를 스크래핑하여 게임 이름에 해당하는 App ID를 찾습니다."""
    
    search_url = f"https://store.steampowered.com/search/?term={game_name}&supportedlang=koreana"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"🔄 Steam 상점 검색 중: '{game_name}'")
    
    try:
        response = requests.get(search_url, headers=headers, timeout=15)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 첫 번째 검색 결과 행 (가장 정확한 결과일 가능성이 높음)
        first_result = soup.find('a', class_='search_result_row')
        
        if first_result:
            # App ID는 HTML 요소의 data-ds-appid 속성에 포함되어 있음
            app_id = first_result.get('data-ds-appid')
            # 게임 제목도 함께 가져와서 확인
            title_tag = first_result.find('span', class_='title')
            game_title = title_tag.text if title_tag else "제목 미상"
            
            if app_id:
                print(f"✅ App ID 발견: {app_id} ({game_title})")
                return int(app_id)
        
        print(f"❌ '{game_name}'에 해당하는 게임 결과를 찾을 수 없습니다.")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Steam 검색 페이지 접근 실패: {e}")
        return None
    except Exception as e:
        print(f"❌ 스크래핑 중 알 수 없는 오류 발생: {e}")
        return None

# ==========================================
# 2단계: App ID로 리뷰 수집

def get_game_reviews(app_id, limit=50):
    """
    특정 게임(app_id)의 한국어 리뷰를 수집합니다. steamreviews가 Pagination을 처리합니다.
    """
    request_params = dict(
        language='koreana', 
        filter='recent',    # 최신순
        num_per_page=100    # 한 번 요청에 가져올 개수
    )
    
    # steamreviews 라이브러리를 통해 데이터 다운로드
    review_dict, _ = steamreviews.download_reviews_for_app_id(
        app_id, 
        chosen_request_params=request_params
    )
    
    reviews_data = []
    if 'reviews' in review_dict:
        # 리뷰 ID를 기준으로 정렬하여 수집 순서를 안정화
        sorted_reviews = sorted(review_dict['reviews'].items(), key=lambda x: x[0], reverse=True) 
        
        count = 0
        for review_id, review in sorted_reviews:
            if count >= limit: break
            
            # 요청하신 모든 필드를 추출하여 저장
            reviews_data.append({
                'review_id': review_id,
                'author_id': review['author']['steamid'],
                'playtime_forever': review['author']['playtime_forever'], # 총 플레이 시간(분)
                'review_text': review['review'],
                'voted_up': review['voted_up'] # 추천/비추천 여부 (True/False)
            })
            count += 1
            
    return reviews_data

# ==========================================
# 메인 실행 함수

def main_crawler():
    game_name = input("🔍 검색할 게임 이름을 입력하세요: ")
    try:
        limit = int(input("🔢 수집할 리뷰 개수를 입력하세요 (예: 200): "))
        if limit < 10: raise ValueError
    except ValueError:
        print("경고: 유효하지 않은 입력입니다. 기본값 100개로 설정합니다.")
        limit = 100

    # 1. Robust Name to ID Lookup
    app_id = get_app_id_by_name(game_name)
    if not app_id:
        print("크롤링을 종료합니다.")
        return

    # 2. Review Crawling
    print(f"\n🚀 '{game_name}' ({app_id})의 리뷰 {limit}개 수집 시작...")
    reviews_data = get_game_reviews(app_id, limit)
    
    # 3. Save Data
    if reviews_data:
        output_filename = f"reviews_{app_id}_{limit}_{game_name.replace(' ', '_')}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(reviews_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n🎉 크롤링 완료: 총 {len(reviews_data)}개 리뷰 수집.")
        print(f"📂 데이터가 '{output_filename}' 파일에 저장되었습니다.")
        
        return reviews_data
    else:
        print("⚠️ 수집된 리뷰가 없습니다. 크롤링 실패 또는 리뷰가 부족합니다.")
        return []

if __name__ == "__main__":
    main_crawler()
    