# 幣安自動交易：VPS 安裝與使用

程式：`binance_trader.py`（`grid_trading/trader/`）。照實盤維加斯 4H + 2H 的規則自動開倉，止損和三個止盈掛在幣安上，
程式只負責移動止損。網站和 Telegram 推播照舊在 GitHub 上跑，這個程式是另外一個服務，兩邊互不影響。

上線順序：**紙上交易（不用金鑰）2 週 → 測試網 2 週 → 小額實盤**。每一段沒問題才進下一段。

## 0. 先把 VPS 弄安全

- 改 root 密碼：`passwd`
- 改用 SSH 金鑰登入（在自己電腦上 `ssh-keygen -t ed25519`，再 `ssh-copy-id root@你的IP`），確定能用金鑰登入後，
  在 `/etc/ssh/sshd_config` 設 `PasswordAuthentication no`，然後 `systemctl restart ssh`。
- 密碼、API 金鑰都不要貼到聊天、GitHub 或任何文件裡。

## 1. 安裝（一次就好）

```bash
apt update && apt install -y git python3 python3-venv
useradd -m -s /bin/bash trader          # 用一般帳號跑，不用 root
su - trader
git clone https://github.com/dennyhty/grid-trading.git
cd grid-trading
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
curl https://fapi.binance.com/fapi/v1/time   # 有回時間數字才繼續
```

## 2. 紙上交易（不用金鑰）

```bash
.venv/bin/python binance_trader.py --mode paper --once   # 先跑一輪看有沒有錯
```

沒錯的話裝成背景服務（回到 root 執行）：

```bash
cp /home/trader/grid-trading/deploy/binance-trader.service /etc/systemd/system/binance-trader@.service
systemctl daemon-reload
systemctl enable --now binance-trader@paper
journalctl -u binance-trader@paper -f      # 看即時紀錄，Ctrl+C 離開
```

要推 Telegram 的話，把 Bot Token 和 Chat ID 寫進 `/home/trader/trader.env`（見第 3 步，同一個檔案）。
紀錄在 `/home/trader/grid-trading/trader_state/paper.json`。

## 3. 測試網

1. 到幣安合約測試網（testnet.binancefuture.com）登入，在 API Key 頁面建一組測試金鑰。這是假錢，和正式帳戶無關。
2. 在 VPS 上寫環境變數檔（只有 trader 帳號讀得到）：

```bash
cat > /home/trader/trader.env <<'EOF'
BINANCE_API_KEY=貼上測試網金鑰
BINANCE_API_SECRET=貼上測試網密鑰
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
chmod 600 /home/trader/trader.env && chown trader /home/trader/trader.env
systemctl stop binance-trader@paper
systemctl enable --now binance-trader@testnet
```

程式第一次下單時會自動把帳戶設成雙向持倉，每個幣設成逐倉。槓桿最多 5 倍，止損越遠越低（強平價至少在止損距離的 2 倍外），
而且不會高過這個幣其他還開著的單允許的倍數（槓桿是整個幣共用的）。逐倉有倉位時幣安不准調低槓桿（-4161），
這時沿用目前的槓桿照樣開倉，並推播警告。

## 4. 正式帳戶（測試網跑過再做）

1. 用專用的合約帳戶（可以的話開子帳戶），帳戶裡**不要有手動倉位**（不是程式開的倉位只提醒、不管理）。
2. 轉 300U 進合約帳戶（先小額，穩了再加）。
3. 查 VPS 的 IP（IPv4 和 IPv6 都查）：
   ```
   curl -4 -s ifconfig.me; echo; curl -6 -s ifconfig.me; echo
   ```
4. 建 API 金鑰：只勾「讀取」和「合約」，**不要勾提幣、萬能劃轉、現貨**；IP 限制填上一步的 IP。
5. 更新程式並重開測試網（測試網照舊讀 `trader.env`）：
   ```
   sudo -u trader bash -c 'cd /home/trader/grid-trading && git pull'
   sudo cp /home/trader/grid-trading/deploy/binance-trader.service /etc/systemd/system/binance-trader@.service
   sudo systemctl daemon-reload && sudo systemctl restart binance-trader@testnet
   ```
6. 實盤的金鑰另放一個檔（服務會在 `trader.env` 之後再讀 `trader-<模式>.env`，後面的蓋掉前面的）：
   ```
   sudo -u trader nano /home/trader/trader-live.env
   ```
   內容：`BINANCE_API_KEY=...`、`BINANCE_API_SECRET=...`、`TRADER_ALLOW_LIVE=1`，想限制動用的資金再加 `TRADER_MAX_EQUITY=300`
   （算倉位的權益最多當 300U；命令列 `--max-equity` 優先）。存檔後 `sudo chmod 600 /home/trader/trader-live.env`。
7. 上線前檢查（唯讀，不下單、不改設定、不推 Telegram）：
   ```
   sudo -u trader bash -c 'cd /home/trader/grid-trading && set -a && . ../trader.env && . ../trader-live.env && set +a && .venv/bin/python binance_trader.py --mode live --check'
   ```
   會列出權益、雙向持倉狀態、現有持倉和掛單、金鑰權限（能提幣或沒限制 IP 會大聲警告）、每筆風險約幾 U。有 `[警告]` 先處理再往下。
8. 啟動：`sudo systemctl enable --now binance-trader@live`。啟動時 Telegram 會推一則「幣安實盤啟動，權益…」。
9. 第一張單成交後，到幣安 App 確認這張單有 1 個止損 + 3 個止盈。

## 日常操作

| 要做的事 | 指令 |
| --- | --- |
| 暫停開新倉（已開的照常管理） | `touch /home/trader/grid-trading/trader_state/pause` |
| 只暫停實盤開新倉 | `touch /home/trader/grid-trading/trader_state/pause-live`（測試網用 `pause-testnet`；`pause` 則全部模式一起暫停） |
| 恢復 | `rm /home/trader/grid-trading/trader_state/pause`（或 `pause-live`） |
| 看紀錄 | `journalctl -u binance-trader@testnet -f`（實盤：`journalctl -u binance-trader@live -f`） |
| 更新程式 | `su - trader -c 'cd grid-trading && git pull'`，再 `systemctl restart binance-trader@testnet` |
| 檢查某個幣在程式和幣安上對不對得上（唯讀） | `su - trader -c 'cd grid-trading && set -a && . /home/trader/trader.env && set +a && .venv/bin/python deploy/trader_diag.py APTUSDT'` |
| 完全停止 | `systemctl stop binance-trader@testnet`（交易所上的止損止盈還在，倉位不會沒保護） |

## 規則

- 進場：只做剛收盤那根的訊號，停機期間錯過的不補。同一個幣同一個週期一次一張；2H、4H 可以同時各一張（同方向也可以），
  在幣安上會合成同一個倉位，但每張單的止損、止盈各用自己的數量掛、各自管理。
- 每筆風險 0.25%（`--risk`）。最小下單量讓實際風險超過 1%（`--max-risk`）的訊號不做並推播原因。
- 流動性門檻（2026-11-17 起）：訊號當下近 30 天日均成交額（成交量 × 收盤價）低於 5,000 萬 U 的幣不做，推播原因（`--min-adv`，0 = 不設）。
  研究見 `reports/signals/report_vegas_improve_20261115.md`：這些幣扣成本後幾乎不賺，2026 年的虧損大多在這裡。網站的訊號照舊全部顯示。
- 掛單（2026-11-17 起，`--taker` 改回市價）：
  - 進場先在訊號收盤價掛只掛單（GTX）的限價單；10 分鐘內（`--entry-wait 600`）沒成交完就取消，剩下的改市價補上。
    價格已經比訊號價好、幣安拒絕只掛單時直接市價進場。限價單成交前沒有倉位，也還沒掛止損；成交後才照實際成交均價掛止損和止盈。
    部分成交時最多再等 60 秒（這段倉位還沒有止損），之後剩下的改市價、馬上掛止損。
    完全沒成交、市價又下不了（例如剩下的不到最小名目金額）就放棄這張，不記帳本。
  - 三個止盈改成掛在 TP 價的一般限價單（GTC）。全部成交才算觸及；部分成交時，等止損、超時平倉或出場時撤單，已成交的那部分照實際價格記帳。
  - 止損照舊是條件市價單（STOP_MARKET，標記價格觸發），一定會出場。
- 出場：止損 1.5 ATR；TP 1R / 3R / 5R 各平 1/4；TP1 後移到保本、TP2 後移到 TP1；TP3 後最後 1/4 用 3 ATR 追蹤
  （TP3 成交當下就從 TP3 價往回 3 ATR 起算，之後每根 K 線收盤更新，只往有利方向移，和回測一樣）；
  240 根還沒結束就市價平倉。價位以實際成交價為準，止損距離照訊號的比例。
- 回撤控制（預設關，Denny 2026-10-20 決定不限張數）：`--max-open 15` 同時最多 15 張；`--dd-half-r 20` 回撤 20R 後風險減半，回到新高才恢復。
  要開就改 `deploy/binance-trader.service` 的 `ExecStart` 加上這兩個參數。
- 止損掛不上就立刻市價平倉；止損單不見了會自動補掛；不是這個程式開的倉位只提醒、不碰。
- 同幣同向兩張單時有個小縫：某張的 TP 成交到程式把止損數量改小之間（最多約 1～2 分鐘），若剛好打到止損，
  可能多平到另一張的一點點數量；止損成交後到程式取消它剩下的止盈之間也一樣。程式發現時會先記 TP 再記止損，
  比對交易所的合併倉位，真的多平了就從另一張單扣掉（照止損成交價記帳）、把它的止損數量改小，並推播。
- 下單不盲目重送：逾時或幣安 5xx 時先用 client id 查單，查得到就當作已下單，查無此單才重送。
  進場單查也查不出結果時推播「下單結果不明」（含幣名、方向、client id），請到幣安檢查有沒有沒止損的倉位。
- 每小時比對一次幣安倉位（一個請求），比程式記錄多就提醒，同一個差異只提醒一次。
- 帳本（狀態檔的 ledger）：平倉時用這張單自己的訂單成交明細算損益、扣實際手續費（`pnl_source: fills`，另存估算值 `pnl_est`）；
  對不上（例如多平到另一張單、手續費用 BNB 付、查詢失敗、舊的單沒有訂單 id）就用觸發價和 0.05% 手續費估算（`pnl_source: estimate`）。
  兩種都沒算資金費。
- 打幣安 API 的頻率：持倉管理每 60 秒一輪（`--poll`），掛單一輪只用一個請求列出所有幣的條件單；紙上交易每張單每 5 分鐘才重算一次，
  收盤後的掃描比測試網晚 75 秒（`--scan-delay`），兩個服務不會在同一秒一起抓 K 線。要更省就把 `--poll` 調大，止損、止盈都掛在交易所上，不受影響。

## 警報、心跳、狀態檔備份

- **警報（Telegram）**：持倉管理（被幣安 418 封鎖、或連續失敗）或整個迴圈連續失敗超過 10 分鐘，推**一次**警報，
  內容有原因，被封時有解封時間（UTC）；恢復時推一次「已恢復」和中斷多久。同一段中斷不重複推，10 分鐘內恢復不推。
  Telegram 不在幣安上，被封期間照樣送得出去。掃描時有幣抓不到 K 線（補抓一次後仍失敗）也會推一次，列出數量和最多 5 個幣名；
  這根照樣記成已處理，不會重掃。
- **推播不擋交易**：訊息先放進佇列，由背景執行緒照順序送；程式結束（`--once`、systemd 停止）前最多等 30 秒送完。
- **心跳**：每輪寫 `trader_state/<mode>.heartbeat.json`（`last_loop`、`last_manage_ok`、`problem`，毫秒時間戳；
  `stream` 是即時推送的狀態、`stream_last_event` 是最後一個推播事件的時間，見下一節）。
  `deploy/trader_feed.py` 把它放進每個 book 的 `heartbeat`（沒有檔案就是 `null`），績效頁顯示「交易程式最後回報」，
  比 feed 產生時間晚 10 分鐘以上或有問題時變警示色。
- **狀態檔備份**：每天（UTC）第一次存檔時寫 `<mode>.json.bak-YYYYMMDD`，留最近 7 份。存檔前先 fsync 暫存檔再改名；內容沒變就不重寫。
- **狀態檔壞了怎麼辦**：
  - 啟動時程式會把壞檔改名成 `<mode>.json.corrupt-時間`（留著查原因），改用最新能讀的備份，並推播告訴你。
    備份是那天第一次存檔的內容，之後開、平的倉可能不在裡面：**馬上到幣安核對倉位**，缺的手動補進狀態檔（先停服務）。
  - 壞檔又沒有能用的備份時，程式推播一次後結束（原檔不動），不會用空狀態開始（否則已開的倉位沒人管，只剩交易所上的止損止盈）。
    處理：停服務 → 從別處還原或手動修好 JSON → 刪掉 `<mode>.json.alerted` → 重開。

## 即時推送（帳戶 websocket）

測試網和實盤會連幣安的 user data stream。成交、止損止盈觸發、倉位變動時幣安會推事件，程式收到就**馬上跑一輪平常的持倉管理**
（等 1 秒把同一批事件合成一輪，兩輪之間至少隔 3 秒）。推播只決定「什麼時候查」：成交、數量、損益照舊全部用 REST 查，
所以推播漏了、晚了、格式改了都不會記錯帳。

- **輪詢照樣是保險**：推送正常時每 300 秒（`--poll-stream`）還是會輪詢一次；推送不正常時照舊每 60 秒（`--poll`）。
  收盤掃描照樣準時（收盤 + `--scan-delay`）。
- **怎樣才算正常**：連著，而且這個網址**收過**訂單 / 條件單 / 帳戶事件（ORDER_TRADE_UPDATE、ALGO_UPDATE、ACCOUNT_UPDATE）。
  只連上不算：舊網址可能連得上卻什麼都不推，和「帳戶剛好沒事」分不出來。所以剛啟動、還沒有任何成交或掛單前都是 60 秒輪詢，
  程式第一次開倉、掛單之後才會切到 300 秒。
- **關掉**：加 `--no-stream`（改 `deploy/binance-trader.service` 的 `ExecStart`），就和以前一樣只輪詢。
- **網址**：幣安 2026 年改了網址格式，細節沒能確認，預設依序試
  `wss://fstream.binance.com/private/ws?listenKey={listenKey}&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE/ALGO_UPDATE`
  和舊的 `wss://fstream.binance.com/ws/{listenKey}`（測試網是 `fstream.binancefuture.com`）。連不上或一連上就斷會換下一個。
  要換網址就在 `trader.env` 加 `BINANCE_WS_URL=範本1,範本2`（逗號分隔，每個都要有 `{listenKey}`）。
- **安裝**：要 `websocket-client` 套件。更新程式後在 venv 裡跑一次
  `su - trader -c 'cd grid-trading && .venv/bin/pip install -r requirements.txt'` 再重開服務。
  沒裝也不會當掉：紀錄印一行「沒有安裝 websocket-client，即時推送關閉」，照舊輪詢。
- **listenKey**：程式自己申請、每 30 分鐘延長，被幣安 418 封鎖期間不送；連滿 23 小時主動重連（幣安 24 小時會切斷）；
  斷線後重連（等 1、2、4…最多 60 秒）。listenKey 等於帳戶推播的鑰匙，紀錄裡一律顯示成 `***`。

**怎麼看有沒有在動**：

- 紀錄（`journalctl -u binance-trader@testnet -f`）：`即時推送已連線（範本 #0：…）` → 第一次有事件時
  `即時推送確認可用：範本 #0 收到 ORDER_TRADE_UPDATE`；斷線時 `即時推送斷線：…原因`（同樣的原因只印一次）。
- 心跳檔的 `stream`：`healthy`（正常，300 秒輪詢）、`connected`（連著但還沒收過事件，60 秒輪詢）、
  `down`（斷線重連中，60 秒輪詢）、`off`（`--no-stream`、紙上交易或沒裝套件）。
- Telegram：啟動後第一次確認可用推一次「即時推送已啟用…」；確認過之後斷超過 10 分鐘推一次警報、恢復推一次，重連不洗版。
