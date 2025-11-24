def main():
    //node --experimental-worker <file>
    async function index_game() {
    const { Worker } = require("worker_threads");
    let startTime = process.uptime(); // 프로세스 시작 시간
    let jobSize = 10;
    let myWorker1, myWorker2, myWorker3

    myWorker1 = new Worker(__dirname + "/all_game_update.js");
    myWorker2 = new Worker(__dirname + "/all_game_update.js");
    myWorker3 = new Worker(__dirname + "/all_game_update.js");
    let endTime = process.uptime();
    console.log("main thread time: " + (endTime - startTime)); // 스레드 생성 시간 + doSomething 처리하는 데 걸린 시간.
    }
    module.exports = index_game;
    index_game();

    const Worker = require("worker_threads");

    let requestGames = async () => {
    let num = Worker.threadId;
    // 스레드 3개, 만약 요청할 갯수가 9999개라면 start를 조절하여 스레드당 1111개씩 3개의 컴퓨터가 분할하여 요청.
    let start = 0;
    // GetAppList/v2 에서 얻어온 게임 appid 리스트를 스레드에 분배
    let { list, start_point } = await finAllList(num, start);
    
    if (!list) {
        console.log(num, "번 스레드 필요없음")
        return;
    }
    // 각 스레드 요청 실시
    await updateAll(list, start_point)
    };


    let finAllList = async (offset, start) => {
    //게임 리스트 확보
    await detail.setTimeoutPromise((offset - 1) * 30000) // 30초에 하나씩 시작
    let res = await request(
        "Get",
        "https://api.steampowered.com/ISteamApps/GetAppList/v2"
    );
    if (res.getBody("utf8") !== undefined) {
        const response = JSON.parse(res.getBody("utf8"));
        if (res.getBody("utf8").slice(0, 6) !== "<HTML>") {
        let apps = response.applist.apps;
        
        // 각 스레드 시작지점 설정
        let start_point = ((offset - 1) * 1111) + start
        const log = `
        ===================================================================
            ${offset}-Worker START!! | 시작: ${start_point} | ${offset < 3 ? "30초 뒤 다음 worker 시작" : "Worker threads 시작 완료"} 
        ===================================================================
                `

        // 마지막 스레드 분기처리
        if (offset === 8) {
            if (start_point < apps.length) {
            if (start_point + 1111 > apps.length) {
                console.log(log)
                return { list: apps.slice(start_point, -1), start_point };
            } else {
                console.log(log)
                return { list: apps.slice(start_point, start_point + 1111), start_point };
            }
            }
            return { list: false, start_point };
        }

        const list = apps.slice(start_point, start_point + 1111)

        console.log(log)
        return { list, start_point };
        } else {
        console.log(res.body.slice(0, 6) + i);
        }
    }
    };

if __name__ == '__main__':
    main()
