// ==========================================
// Sample Code임으로 이 프로젝트에서는 사용하지 않을 예정입니다.
// 출처: https://velog.io/@kjjdsa/%EC%8A%A4%ED%8C%80-%EA%B2%8C%EC%9E%84%EA%B3%BC-%EB%A6%AC%EB%B7%B0-%ED%81%AC%EB%A1%A4%EB%A7%81
// 위 출처 코드를 변형하여 작성하였음
// ==========================================


const axios = require("axios");
const fs = require("fs");

// ==========================================
// 설정 (Settings)
// ==========================================
const DELAY_MS = 1500; // 요청 간 딜레이 (1.5초 권장 - 스팀 차단 방지)
const REVIEWS_PER_PAGE = 100; // Steam API 최대값
const OUTPUT_FILE = "collected_reviews.json";

// Axios 클라이언트 생성
const client = axios.create({
  timeout: 60000,
  headers: {
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip,deflate,compress",
  },
});

// ==========================================
// 유틸리티 함수
// ==========================================
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ==========================================
// 핵심 로직
// ==========================================

/**
 * 1. 게임 이름으로 App ID를 검색합니다.
 * @param {string} gameName 검색할 게임 이름
 * @returns {number|null} App ID 또는 null
 */
async function getAppIdByName(gameName) {
  console.log(`🔄 '${gameName}'에 해당하는 App ID 검색 중...`);
  try {
    const res = await client.get(
      "https://api.steampowered.com/ISteamApps/GetAppList/v2"
    );
    const apps = res.data.applist.apps;
    
    // 이름 매칭 (대소문자 구분 없이, 정확히 일치하는 항목 검색)
    const normalizedName = gameName.trim().toLowerCase();
    
    const matchedApp = apps.find(app => 
        app.name && app.name.toLowerCase().includes(normalizedName)
    );
    
    if (matchedApp) {
      console.log(`✅ App ID 발견: ${matchedApp.appid} (${matchedApp.name})`);
      return matchedApp.appid;
    }
    
    console.log("❌ App ID를 찾을 수 없습니다. 이름을 정확히 입력해 주세요.");
    return null;

  } catch (error) {
    console.error("❌ 앱 리스트 가져오기 실패:", error.message);
    return null;
  }
}

/**
 * 2. App ID를 사용하여 원하는 개수만큼 리뷰를 가져옵니다. (Pagination 포함)
 * - review.voted_up (추천 여부)
 * - review.author.steamid (사용자 식별 ID)
 * - review.author.playtime_forever (총 플레이 시간)
 * @param {number} appid 게임 App ID
 * @param {number} limit 수집할 리뷰의 목표 개수
 * @returns {Array} 수집된 리뷰 리스트
 */
async function getAppReviews(appid, limit) {
  let allReviews = [];
  let cursor = "*"; 
  let totalCollected = 0;

  console.log(`🚀 리뷰 수집 시작 (목표: ${limit}개)`);

  while (totalCollected < limit) {
    const remaining = limit - totalCollected;
    const count = Math.min(remaining, 100); // 100은 REVIEWS_PER_PAGE 상수 사용 가능

    try {
      const url = `https://store.steampowered.com/appreviews/${appid}?json=1&language=koreana&filter=recent&review_type=all&num_per_page=${count}&cursor=${cursor}`;
      const res = await client.get(url);
      
      if (res.data.success !== 1 || !res.data.reviews || res.data.reviews.length === 0) {
        console.log("⚠️ 더 이상 리뷰가 없거나 요청에 실패했습니다.");
        break;
      }

      // **핵심 수정: 데이터 매핑 및 추출**
      const newReviews = res.data.reviews.map(review => ({
          review_text: review.review,

          voted_up: review.voted_up, // true/false (추천 여부)
          author_id: review.author.steamid, // 사용자 식별 ID (Steam ID)
          playtime_forever: review.author.playtime_forever, // 총 플레이 시간 (단위: 분)
          timestamp_created: review.timestamp_created, // 리뷰 작성 시간 (참고용)
          playtime_at_review: review.author.playtime_at_review // 리뷰 작성 시점 플레이 시간
      }));

      allReviews.push(...newReviews);
      totalCollected += newReviews.length;
      cursor = res.data.cursor; // 다음 페이지 커서
      
      console.log(`  - 수집 완료: ${totalCollected}개 (이번 페이지: ${newReviews.length}개)`);
      
    } catch (error) {
      console.error(`❌ 리뷰 요청 실패: ${error.message}. 잠시 대기 후 재시도...`);
      await sleep(DELAY_MS * 3);
      continue;
    }
    
    if (totalCollected < limit) {
        await sleep(DELAY_MS); 
    }
  }

  console.log(`✅ 리뷰 수집 완료. 최종 개수: ${allReviews.length}개`);
  return allReviews;
}

/**
 * 3. 메인 실행 함수
 */
async function main() {
    // 커맨드 라인 인수를 받음 (node crawler_V2.js "게임이름" 리뷰개수)
    const gameName = process.argv[2];
    const limit = parseInt(process.argv[3]) || 10; // 기본 10개

    if (!gameName) {
        console.error("⛔ 사용법: node crawler_V2.js \"게임 이름\" [리뷰 개수]");
        console.log("예시: node crawler_V2.js \"발더스 게이트 3\" 50");
        return;
    }

    // 1단계: App ID 찾기
    const appid = await getAppIdByName(gameName);
    if (!appid) return;

    // 2단계: 리뷰 크롤링
    const reviews = await getAppReviews(appid, limit);
    
    // 3단계: 결과 저장
    const finalData = {
        appid: appid,
        game_name: gameName,
        review_count: reviews.length,
        reviews: reviews
    };

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(finalData, null, 2));
    console.log(`\n !최종 데이터가 ${OUTPUT_FILE}에 저장되었습니다.`);
}

main();
