
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(layout="wide")
kosdaq_tickers = {
    '에코프로': '086520.KQ',
    '에코프로비엠': '247540.KQ',
    '엘앤에프': '066970.KQ',
    '씨아이에스': '222080.KQ',
    '포스코DX': '022100.KQ'
}

st.title("[코스닥 급등 신호 실시간 분석] (모바일 지원)")
symbol = st.selectbox("분석할 종목을 선택하세요:", list(kosdaq_tickers.keys()))
ticker = kosdaq_tickers[symbol]

def get_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def safe_float(val):
    try:
        if isinstance(val, pd.Series):
            val = val.iloc[0] if not val.empty else None
        if isinstance(val, (int, float, np.number)) and not np.isnan(val):
            return float(val)
    except:
        pass
    return None

def compute_signals(df):
    df['RSI_14'] = get_rsi(df['Close'], 14)
    df['RSI_9'] = get_rsi(df['Close'], 9)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()

    price_up = df['Close'] > df['Close'].shift(1)
    direction = price_up.astype(int).replace({0: -1})
    df['OBV'] = (df['Volume'] * direction).fillna(0).cumsum()
    df['OBV_20'] = df['OBV'].rolling(20).mean()

    signals = []
    for i in range(2, len(df)):
        win = df.iloc[i-2:i+1]

        rsi_ok = False
        for j in range(1, 3):
            rsi_9_now = safe_float(win.iloc[j]['RSI_9'])
            rsi_14_now = safe_float(win.iloc[j]['RSI_14'])
            rsi_9_prev = safe_float(win.iloc[j-1]['RSI_9'])
            rsi_14_prev = safe_float(win.iloc[j-1]['RSI_14'])
            if None not in (rsi_9_now, rsi_14_now, rsi_9_prev, rsi_14_prev):
                if rsi_9_now > rsi_14_now and rsi_9_prev <= rsi_14_prev:
                    rsi_ok = True
                    break

        obv_ok = False
        for _, row in win.iterrows():
            obv = safe_float(row['OBV'])
            obv20 = safe_float(row['OBV_20'])
            if None not in (obv, obv20) and obv > obv20:
                obv_ok = True
                break

        ma_ok = False
        for j in range(1, 3):
            prev_close = safe_float(win.iloc[j-1]['Close'])
            curr_close = safe_float(win.iloc[j]['Close'])
            prev_ma5 = safe_float(win.iloc[j-1]['MA5'])
            prev_ma20 = safe_float(win.iloc[j-1]['MA20'])
            curr_ma5 = safe_float(win.iloc[j]['MA5'])
            curr_ma20 = safe_float(win.iloc[j]['MA20'])

            vals = [prev_close, curr_close, prev_ma5, prev_ma20, curr_ma5, curr_ma20]
            if all(v is not None for v in vals):
                cond1 = bool(prev_close <= prev_ma5)
                cond2 = bool(prev_close <= prev_ma20)
                cond3 = bool(curr_close > curr_ma5)
                cond4 = bool(curr_close > curr_ma20)

                if (cond1 or cond2) and (cond3 and cond4):
                    ma_ok = True
                    break

        rsi_val = safe_float(df.iloc[i]['RSI_14'])
        obv_val = safe_float(df.iloc[i]['OBV'])
        obv20_val = safe_float(df.iloc[i]['OBV_20'])
        is_premium = rsi_val is not None and 50 <= rsi_val <= 65

        if rsi_ok and obv_ok and ma_ok:
            signals.append({
                '날짜': df.index[i].strftime('%Y-%m-%d'),
                '종가': round(safe_float(df.iloc[i]['Close']), 2),
                'RSI(14)': round(rsi_val, 2) if rsi_val is not None else None,
                'OBV': round(obv_val, 0) if obv_val is not None else None,
                'OBV_20': round(obv20_val, 0) if obv20_val is not None else None,
                '프리미엄': is_premium
            })
    return pd.DataFrame(signals)

st.info(f"{symbol} 종목의 6개월 데이터를 불러오는 중입니다...")
df = yf.download(ticker, period='6mo', interval='1d', auto_adjust=True)

if df is None or df.empty:
    st.error("데이터를 불러오지 못했습니다.")
else:
    df.reset_index(inplace=True)
    signal_df = compute_signals(df)
    st.success(f"총 {len(signal_df)}건의 급등 후보 발생!")
    st.subheader("**프리미엄 신호 (RSI 50~65)**")
    st.dataframe(signal_df[signal_df['프리미엄']], use_container_width=True)
    st.subheader("**일반 신호**")
    st.dataframe(signal_df[~signal_df['프리미엄']], use_container_width=True)
