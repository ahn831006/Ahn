
# Streamlit 급등 신호 대시보드 (중복 제거 없이, 완화된 조건 적용)
import streamlit as st
import pandas as pd
import numpy as np
import datetime

@st.cache_data
def load_data():
    df = pd.read_csv("한국첨단소재_data.csv", encoding='cp949')
    df = df.rename(columns={'일자': 'Date', '종가': 'Close', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'})
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def compute_indicators(df):
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['RSI_9'] = compute_rsi(df['Close'], 9)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['OBV'] = compute_obv(df['Close'], df['Volume'])
    df['OBV_20'] = df['OBV'].rolling(20).mean()
    return df

def detect_signals(df):
    signals = []
    for i in range(2, len(df)):
        if i < 20:
            continue

        win = df.iloc[i-2:i+1]
        rsi_ok = any(
            (win.iloc[j]['RSI_9'] > win.iloc[j]['RSI']) and
            (win.iloc[j-1]['RSI_9'] <= win.iloc[j-1]['RSI']) for j in range(1, 3)
        )
        obv_ok = any(win['OBV'] > win['OBV_20'])

        ma_ok = False
        for j in range(1, 3):
            prev_close = win.iloc[j-1]['Close']
            curr_close = win.iloc[j]['Close']
            prev_ma5 = win.iloc[j-1]['MA5']
            prev_ma20 = win.iloc[j-1]['MA20']
            curr_ma5 = win.iloc[j]['MA5']
            curr_ma20 = win.iloc[j]['MA20']
            cross_ma5 = (prev_close <= prev_ma5) and (curr_close > curr_ma5)
            cross_ma20 = (prev_close <= prev_ma20) and (curr_close > curr_ma20)
            if cross_ma5 or cross_ma20:
                ma_ok = True
                break

        if rsi_ok and obv_ok and ma_ok:
            is_premium = 50 <= df.iloc[i]['RSI'] <= 65
            signals.append({
                '신호일': df.iloc[i]['Date'],
                '종가': df.iloc[i]['Close'],
                'RSI': round(df.iloc[i]['RSI'], 2),
                'OBV': df.iloc[i]['OBV'],
                'OBV_20': df.iloc[i]['OBV_20'],
                'Premium': is_premium
            })
    return pd.DataFrame(signals).drop_duplicates()

# 메인 실행
st.title("급등 신호 실시간 대시보드 (완화된 조건)")
df = load_data()
df = compute_indicators(df)
signal_df = detect_signals(df)

st.success(f"총 {len(signal_df)}개의 급등 후보 발견")

premium_df = signal_df[signal_df['Premium']]
normal_df = signal_df[~signal_df['Premium']]

st.subheader("프리미엄 신호 (RSI 50~65)")
st.dataframe(premium_df)

st.subheader("일반 신호")
st.dataframe(normal_df)

import streamlit.components.v1 as components
components.html("""
<script>
  setTimeout(function() { window.location.reload(); }, 60000);
</script>
""", height=0)
