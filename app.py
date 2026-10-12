import io
import os
import re
import requests
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from streamlit_gsheets import GSheetsConnection
from functools import lru_cache

st.set_page_config(page_title="台股集保籌碼追蹤 Pro", layout="wide")

# -------------------------------------------------------------
# 1. 繁體中文字型載入 (防崩潰安全設置)
# -------------------------------------------------------------
my_font = None
system_cjk_fonts = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
]

installed_font = None
for fpath in system_cjk_fonts:
    if os.path.exists(fpath):
        installed_font = fpath
        break

if installed_font:
    try:
        fm.fontManager.addfont(installed_font)
        my_font = fm.FontProperties(fname=installed_font)
        plt.rcParams['font.family'] = my_font.get_name()
    except Exception:
        pass

plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------
# 2. Google Sheets 純讀取設定 (免 Service Account 密鑰)
# -------------------------------------------------------------
conn = st.connection("gsheets", type=GSheetsConnection)

def load_google_sheet_data():
    """從公開 Google Sheets 讀取過往歷史集保紀錄"""
    try:
        df = conn.read(ttl=300)
        if df is not None and not df.empty:
            df["資料日期"] = df["資料日期"].astype(str).str.strip()
            df["證券代號"] = df["證券代號"].astype(str).str.strip()
            df["持股分級"] = pd.to_numeric(df["持股分級"], errors="coerce").fillna(0).astype(int)
            df["人數"] = pd.to_numeric(df["人數"], errors="coerce").fillna(0).astype(int)
            df["股數"] = pd.to_numeric(df["股數"], errors="coerce").fillna(0).astype(int)
            df["佔集保庫存數比例%"] = pd.to_numeric(df["佔集保庫存數比例%"], errors="coerce").fillna(0.0)
            return df
    except Exception as e:
        st.warning(f"Google Sheets 讀取提示: {e}")
    return pd.DataFrame()

# -------------------------------------------------------------
# 3. 股票名稱快取 (上市 + 上櫃 全市場自動解析)
# -------------------------------------------------------------
COMMON_STOCK_NAMES = {
    "2330": "台積電", "2454": "聯發科", "3037": "欣興", "2317": "鴻海",
    "2308": "台達電", "2881": "富邦金", "2882": "國泰金", "2603": "長榮",
    "2609": "陽明",   "2615": "萬海",   "3231": "緯創", "2382": "廣達",
    "2327": "國巨",   "3324": "雙鴻",   "6488": "環球晶", "3081": "聯亞",
    "6533": "晶心科", "8299": "群聯",   "2357": "華碩", "3443": "創意",
    "3374": "精材"
}

@st.cache_data(ttl=86400)
def fetch_all_stock_names():
    stock_map = dict(COMMON_STOCK_NAMES)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r_twse = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=4)
        if r_twse.status_code == 200:
            for row in r_twse.json():
                c, n = row.get("Code", "").strip(), row.get("Name", "").strip()
                if c and n: stock_map[c] = n
    except Exception:
        pass
    try:
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        r_tpex = requests.get(tpex_url, headers=headers, timeout=4)
        if r_tpex.status_code == 200:
            for row in r_tpex.json():
                c, n = row.get("SecuritiesCompanyCode", "").strip(), row.get("CompanyName", "").strip()
                if c and n: stock_map[c] = n
    except Exception:
        pass
    return stock_map

def get_stock_name(code):
    code = str(code).strip()
    if code in COMMON_STOCK_NAMES:
        return COMMON_STOCK_NAMES[code]
    stock_map = fetch_all_stock_names()
    return stock_map.get(code, f"台股 {code}")

@lru_cache(maxsize=512)
def get_stock_info(code):
    code = str(code).strip()
    name = get_stock_name(code)
    price = None
    for suffix in [".TW", ".TWO"]:
        try:
            ticker = yf.Ticker(f"{code}{suffix}")
            hist = ticker.history(period="1d", timeout=2)
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                break
        except Exception:
            pass
    return price, name

def get_dynamic_thresholds(price):
    if price is not None and price >= 1000:
        whale_levels = [12, 13, 14, 15]
        retail_levels = list(range(1, 11))
        return whale_levels, retail_levels, "> 400 張", "<= 100 張"
    else:
        whale_levels = [15]
        retail_levels = list(range(1, 12))
        return whale_levels, retail_levels, "> 1000 張", "<= 400 張"

# -------------------------------------------------------------
# 4. TDCC 集保當週資料抓取與指標計算
# -------------------------------------------------------------
def fetch_latest_tdcc_data():
    url = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    df.columns = [str(c).strip().replace("\ufeff", "").replace(" ", "") for c in df.columns]

    col_mapping = {}
    for col in df.columns:
        if "日期" in col: col_mapping["date"] = col
        elif any(k in col for k in ["證券", "代號", "股票"]): col_mapping["code"] = col
        elif any(k in col for k in ["分級", "級距"]): col_mapping["level"] = col
        elif "人數" in col: col_mapping["people"] = col
        elif "股數" in col: col_mapping["shares"] = col
        elif any(k in col for k in ["比例", "佔比"]): col_mapping["ratio"] = col

    std_df = pd.DataFrame()
    std_df["資料日期"] = df[col_mapping["date"]].str.strip()
    std_df["證券代號"] = df[col_mapping["code"]].str.strip()
    std_df["持股分級"] = pd.to_numeric(df[col_mapping["level"]], errors="coerce").fillna(0).astype(int)
    std_df["人數"] = pd.to_numeric(df[col_mapping["people"]], errors="coerce").fillna(0).astype(int)
    std_df["股數"] = pd.to_numeric(df[col_mapping["shares"]], errors="coerce").fillna(0).astype(int)
    std_df["佔集保庫存數比例%"] = pd.to_numeric(df[col_mapping["ratio"]], errors="coerce").fillna(0.0)

    return std_df

def calculate_custom_metrics(df_source, target_date, code, whale_levels, retail_levels):
    sub = df_source[(df_source["資料日期"] == str(target_date)) & (df_source["證券代號"] == str(code))]
    if sub.empty:
        return None

    whale = sub[sub["持股分級"].isin(whale_levels)]
    retail = sub[sub["持股分級"].isin(retail_levels)]
    
    total_shareholders = sub[sub["持股分級"] == 17]["人數"].sum()
    if total_shareholders == 0:
        total_shareholders = sub["人數"].sum()

    return {
        "whale_ratio": whale["佔集保庫存數比例%"].sum(),
        "whale_count": int(whale["人數"].sum()),
        "retail_ratio": retail["佔集保庫存數比例%"].sum(),
        "retail_count": int(retail["人數"].sum()),
        "total_count": int(total_shareholders)
    }

# -------------------------------------------------------------
# 5. 記憶上次搜尋的股號 (Session State + Query Params 雙重記憶)
# -------------------------------------------------------------
# 優先順序：URL 參數 > Session State > 預設值
default_stocks = "2330 2454 3037"
url_stocks = st.query_params.get("stocks", None)

if url_stocks:
    initial_stock_val = url_stocks
elif "saved_stocks" in st.session_state:
    initial_stock_val = st.session_state["saved_stocks"]
else:
    initial_stock_val = default_stocks

# -------------------------------------------------------------
# 6. 主畫面與查詢繪圖流程
# -------------------------------------------------------------
st.title("台股集保大戶/散戶/總股東【批次查詢】與【三合一整合圖表】系統 Pro")

stock_input = st.text_input("請輸入股票代碼 (支援多檔，用空格隔開)", value=initial_stock_val)

if st.button("批次分析與產生圖表", type="primary"):
    # 儲存本次輸入的股票代碼至狀態與 URL
    clean_input = stock_input.strip()
    st.session_state["saved_stocks"] = clean_input
    st.query_params["stocks"] = clean_input

    raw_codes = re.split(r'[\s,，]+', clean_input)
    stock_codes = [c.strip() for c in raw_codes if c.strip()]

    if not stock_codes:
        st.warning("請輸入至少一個合法的股票代碼！")
    else:
        with st.spinner("正在讀取歷史紀錄與抓取集保數據..."):
            try:
                # 1. 讀取 Google Sheets 歷史資料庫 (純讀取，安全快速)
                history_df = load_google_sheet_data()

                # 2. 抓取集保當週最新即時資料
                latest_df = fetch_latest_tdcc_data()
                this_week = str(latest_df["資料日期"].iloc[0])

                # 3. 記憶體中合併歷史與當週資料
                if not history_df.empty:
                    combined = pd.concat([history_df, latest_df], ignore_index=True)
                else:
                    combined = latest_df

                combined = combined.drop_duplicates(subset=["資料日期", "證券代號", "持股分級"])
                all_dates = sorted(combined["資料日期"].unique(), reverse=True)

                combined_results = []
                n_stocks = len(stock_codes)
                fig_combined, axes = plt.subplots(n_stocks, 1, figsize=(11, 5 * n_stocks), squeeze=False)

                for idx, stock_code in enumerate(stock_codes):
                    price, stock_name = get_stock_info(stock_code)
                    whale_levels, retail_levels, w_desc, r_desc = get_dynamic_thresholds(price)

                    cur = calculate_custom_metrics(combined, this_week, stock_code, whale_levels, retail_levels)
                    if not cur:
                        st.error(f"查無代號 {stock_code} 的資料")
                        continue

                    # 抓取最近 5 週歷史資料連線
                    trend_history = {"date": [], "whale_count": [], "retail_count": [], "total_count": []}
                    for d in reversed(all_dates[:5]):
                        m = calculate_custom_metrics(combined, d, stock_code, whale_levels, retail_levels)
                        if m:
                            trend_history["date"].append(d)
                            trend_history["whale_count"].append(m["whale_count"])
                            trend_history["retail_count"].append(m["retail_count"])
                            trend_history["total_count"].append(m["total_count"])

                    if len(trend_history["date"]) > 0:
                        ax1 = axes[idx, 0]
                        dates = trend_history["date"]
                        w_counts = trend_history["whale_count"]
                        r_counts = trend_history["retail_count"]
                        t_counts = trend_history["total_count"]

                        # 左 Y 軸 (總股東、散戶)
                        line1 = ax1.plot(dates, t_counts, marker="^", color="blue", linewidth=2, label="總股東")
                        line2 = ax1.plot(dates, r_counts, marker="s", color="green", linewidth=2, label="散戶")
                        ax1.set_ylabel("總股東 / 散戶 (人)", color="blue", fontproperties=my_font)
                        ax1.tick_params(axis="y", labelcolor="blue")
                        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
                        
                        for i, txt in enumerate(t_counts):
                            ax1.annotate(f"{txt:,}", (dates[i], t_counts[i]), textcoords="offset points", xytext=(0, 8), ha="center", color="blue", fontsize=9, fontproperties=my_font)
                        for i, txt in enumerate(r_counts):
                            ax1.annotate(f"{txt:,}", (dates[i], r_counts[i]), textcoords="offset points", xytext=(0, -14), ha="center", color="green", fontsize=9, fontproperties=my_font)

                        # 右 Y 軸 (大戶)
                        ax2 = ax1.twinx()
                        line3 = ax2.plot(dates, w_counts, marker="o", color="red", linewidth=2, linestyle="--", label="大戶")
                        ax2.set_ylabel("大戶 (人)", color="red", fontproperties=my_font)
                        ax2.tick_params(axis="y", labelcolor="red")
                        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
                        
                        for i, txt in enumerate(w_counts):
                            ax2.annotate(f"{txt:,}", (dates[i], w_counts[i]), textcoords="offset points", xytext=(0, 8), ha="center", color="red", fontsize=9, fontproperties=my_font)

                        lines = line1 + line2 + line3
                        labels = [l.get_label() for l in lines]
                        ax1.legend(lines, labels, loc="upper left", prop=my_font)
                        
                        title_str = f"{stock_name} ({stock_code}) - 人數雙軸趨勢圖 (大戶: {w_desc}, 散戶: {r_desc})"
                        ax1.set_title(title_str, fontsize=13, fontweight="bold", fontproperties=my_font)
                        ax1.grid(True, linestyle="--", alpha=0.3)

                    combined_results.append({
                        "代號": stock_code,
                        "股票名稱": stock_name,
                        "基準週(最新)": this_week,
                        "最新股價": f"${price:.1f}" if price else "-",
                        "大戶持股%": f"{cur['whale_ratio']:.2f}%",
                        "大戶人數": f"{cur['whale_count']:,}人",
                        "散戶持股%": f"{cur['retail_ratio']:.2f}%",
                        "散戶人數": f"{cur['retail_count']:,}人"
                    })

                st.success(f"最新集保日期: {this_week}")
                st.subheader("批次彙整分析報表")
                st.dataframe(pd.DataFrame(combined_results), use_container_width=True)

                st.subheader("趨勢圖 (三合一雙軸整合圖表)")
                plt.tight_layout()
                st.pyplot(fig_combined)

            except Exception as e:
                st.error(f"系統發生錯誤: {str(e)}")
