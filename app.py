import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit 앱 설정
st.title("📈 이동평균 + RSI + MACD + 볼린저밴드 + 거래량 + 모멘텀 전략 백테스트")

# 사용자 입력 받기
symbol = st.text_input("종목 티커 입력 (예: AAPL, TSLA, 005930.KQ)", value="AAPL")
start_date = st.date_input("시작 날짜", pd.to_datetime("2020-01-01"))
end_date = st.date_input("종료 날짜", pd.to_datetime("2024-12-31"))
short_window = st.number_input("단기 이동평균 (Short MA)", min_value=5, max_value=100, value=20)
long_window = st.number_input("장기 이동평균 (Long MA)", min_value=10, max_value=200, value=60)
rsi_threshold = st.slider("RSI 매수 임계값 (미만)", min_value=10, max_value=50, value=30)
macd_enabled = st.checkbox("MACD 시그널 포함", value=True)
bollinger_enabled = st.checkbox("볼린저 밴드 하단 포함 (주가 < 하단선)", value=False)
volume_enabled = st.checkbox("20일 평균 대비 거래량 1.5배 이상 포함", value=False)
momentum_enabled = st.checkbox("최근 10일 수익률 > 0 포함", value=False)

if st.button("🔍 전략 실행"):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)

        # 이동평균
        data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
        data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

        # RSI 계산
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD 계산
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # 볼린저 밴드 계산
        data['BB_Mid'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']

        # 거래량 평균 대비 비교
        data['Volume_Avg'] = data['Volume'].rolling(window=20).mean()

        # 모멘텀 (10일 수익률)
        data['Momentum_10'] = data['Close'].pct_change(periods=10)

        # NaN 제거 (모든 지표 계산 후)
        data = data.dropna().copy()

        # 전략 시그널: 조건 조합
        data["Signal"] = 0
        condition = (data["Short_MA"] > data["Long_MA"]) & (data["RSI"] < rsi_threshold)

        if macd_enabled:
            macd, signal = data["MACD"].align(data["Signal_Line"], join="inner")
            condition = condition & (macd > signal)

        if bollinger_enabled:
            close, lower = data["Close"].align(data["BB_Lower"], join="inner")
            condition = condition & (close < lower)

        if volume_enabled:
            vol, vol_avg = data["Volume"].align(data["Volume_Avg"], join="inner")
            condition = condition & (vol > 1.5 * vol_avg)

        if momentum_enabled:
            momentum = data["Momentum_10"].reindex(condition.index).fillna(0)
            condition = condition & (momentum > 0)

        condition = condition.fillna(False)
        data.loc[condition[condition].index, "Signal"] = 1
        data["Position"] = data["Signal"].diff()

        # 수익률 계산
        data["Market Return"] = data["Close"].pct_change()
        data["Strategy Return"] = data["Market Return"] * data["Signal"].shift(1)
        data["Cumulative Market Return"] = (1 + data["Market Return"]).cumprod()
        data["Cumulative Strategy Return"] = (1 + data["Strategy Return"]).cumprod()

        # 그래프
        st.subheader("누적 수익률 비교")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data["Cumulative Market Return"], label="Market")
        ax.plot(data.index, data["Cumulative Strategy Return"], label="Strategy")
        ax.set_title(f"{symbol} - 커스터마이징 전략")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # 요약
        summary = {
            "종목": symbol,
            "시작일": start_date,
            "종료일": end_date,
            "단기 MA": short_window,
            "장기 MA": long_window,
            "RSI 임계값": rsi_threshold,
            "MACD 포함": macd_enabled,
            "볼린저 밴드 사용": bollinger_enabled,
            "거래량 급증 포함": volume_enabled,
            "모멘텀 조건 포함": momentum_enabled,
            "시장 누적 수익률": f"{(data['Cumulative Market Return'].iloc[-1] - 1):.2%}",
            "전략 누적 수익률": f"{(data['Cumulative Strategy Return'].iloc[-1] - 1):.2%}",
            "최대 낙폭 (MDD)": f"{(data['Cumulative Strategy Return'].cummax() - data['Cumulative Strategy Return']).max():.2%}",
        }

        st.subheader("전략 요약")
        st.dataframe(pd.DataFrame.from_dict(summary, orient="index", columns=["값"]))

        # 보조 지표 시각화
        st.subheader("보조 지표 차트")
        fig2, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(data.index, data['RSI'], label='RSI', color='purple')
        axs[0].axhline(30, color='red', linestyle='--', alpha=0.5)
        axs[0].axhline(70, color='green', linestyle='--', alpha=0.5)
        axs[0].set_title('RSI')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(data.index, data['MACD'], label='MACD', color='blue')
        axs[1].plot(data.index, data['Signal_Line'], label='Signal Line', color='orange')
        axs[1].set_title('MACD')
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(data.index, data['Close'], label='Close Price', color='black')
        axs[2].plot(data.index, data['BB_Upper'], label='BB Upper', color='green', linestyle='--')
        axs[2].plot(data.index, data['BB_Lower'], label='BB Lower', color='red', linestyle='--')
        axs[2].fill_between(data.index, data['BB_Lower'], data['BB_Upper'], color='gray', alpha=0.1)
        axs[2].set_title('Bollinger Bands')
        axs[2].legend()
        axs[2].grid(True)

        st.pyplot(fig2)

    except Exception as e:
        st.error(f"에러 발생: {e}")
