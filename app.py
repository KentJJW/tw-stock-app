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
from functools import lru_cache

st.set_page_config(page_title="台股集保籌碼追蹤 Pro", layout="wide")

# -------------------------------------------------------------
# 1. 繁體中文字型下載與註冊 (自動配置思源黑體)
# -------------------------------------------------------------
font_dir = "./fonts"
os.makedirs(font_dir, exist_ok=True)
font_path = os.path.join(font_dir, "NotoSansTC-Regular.ttf")

if not os.path.exists(font_path) or os.path.getsize(font_path) < 1000:
    try:
        ttf_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
        r = requests.get(ttf_url, timeout=30)
        with open(font_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        st.warning(f"字型下載失敗: {e}")

my_font = None
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    my_font = font_prop

plt.rcParams['axes.unicode_minus'] = False

# -------------------------------------------------------------
# 2. 資料儲存路徑
# -------------------------------------------------------------
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_STORE_PATH = os.path.join(DATA_DIR, "tdcc_history.csv")

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

_GLOBAL_STOCK_MAP = None

def fetch_all_stock_names():
    """一次性快速獲取上市與上櫃全部股票中文名稱"""
    stock_map = dict(COMMON_STOCK_NAMES)
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. 抓取上市股票清單 (TWSE 證交所 API)
    try:
        r_twse = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=4)
        if r_twse.status_code == 200:
            for row in r_twse.json():
                c, n = row.get("Code", "").strip(), row.get("Name", "").strip()
                if c and n:
                    stock_map[c] = n
    except Exception:
        pass

    # 2. 抓取上櫃股票清單 (TPEX 櫃買中心 API，如 3374 精材)
    try:
        tpex_url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        r_tpex = requests.get(tpex_url, headers=headers, timeout=4)
        if r_tpex.status_code == 200:
            for row in r_tpex.json():
                c, n = row.get("SecuritiesCompanyCode", "").strip(), row.get("CompanyName", "").strip()
                if c and n:
                    stock_map[c] = n
    except Exception:
        pass

    return stock_map

def get_stock_name(code):
    global _GLOBAL_STOCK_MAP
    code = str(code).strip()
    
    # 常用快取直接秒出
    if code in COMMON_STOCK_NAMES:
        return COMMON_STOCK_NAMES[code]
    
    # 首次查詢非常用股時，自動載入全市場清單
    if _GLOBAL_STOCK_MAP is None:
        _GLOBAL_STOCK_MAP = fetch_all_stock_names()

    if code in _GLOBAL_STOCK_MAP:
        return _GLOBAL_STOCK_MAP[code]
        
    return f"台股 {code}"

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

# -------------------------------------------------------------
# 4. 集保分級標準定義 (修正 400 張門檻包含 12~15 級)
# -------------------------------------------------------------
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
# 5. TDCC 集保資料抓取與指標計算
# -------------------------------------------------------------
def fetch_latest_tdcc_data():
    url = "https://opendata.tdcc.com.tw/getOD.ashx?id=1-5"
    headers = {"User-Agent": "Mozilla/5.0"}
