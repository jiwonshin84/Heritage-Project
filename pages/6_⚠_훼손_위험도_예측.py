import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# ============================================================
# 0. 페이지 설정 및 디자인
# ============================================================
st.set_page_config(page_title="영천 문화재 훼손 위험 예측", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ 영천 지역 문화재 훼손 위험 실시간 예측")
st.markdown("기상청 및 대기오염 공공데이터를 기반으로 현재 시점의 위험도를 분석합니다.")

# ============================================================
# 1. 데이터 수집 및 전처리 (최신 데이터 위주)
# ============================================================
@st.cache_data(ttl=3600) # 1시간마다 캐시 갱신
def get_latest_environment_data():
    # 기상 데이터 수집 (영천 관측소: 281)
    ASOS_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    # 최근 7일치 기상 수집 (변화율 계산을 위함)
    def fetch_weather():
        url = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
        params = {
            "serviceKey": ASOS_SERVICE_KEY, "numOfRows": "10", "dataType": "JSON",
            "dataCd": "ASOS", "dateCd": "DAY", 
            "startDt": yesterday.strftime('%Y%m%d'), 
            "endDt": today.strftime('%Y%m%d'), "stnIds": "281"
        }
        try:
            res = requests.get(url, params=params).json()
            items = res["response"]["body"]["items"]["item"]
            return pd.DataFrame(items)
        except: return pd.DataFrame()

    weather_df = fetch_weather()
    # 컬럼명 변경 및 타입 변환
    weather_df = weather_df[["tm", "avgTa", "avgRhm", "sumRn", "avgWs"]].copy()
    weather_df.columns = ["date", "temp", "humidity", "rainfall", "wind"]
    for col in ["temp", "humidity", "rainfall", "wind"]:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
    weather_df["rainfall"] = weather_df["rainfall"].fillna(0)
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    # 미세먼지/대기오염 데이터 (사용자 지정 경로)
    air_path = "data/processed/[2019_2025] air_quality.csv"
    if os.path.exists(air_path):
        air_df = pd.read_csv(air_path)
        air_df["date"] = pd.to_datetime(air_df["date"])
        # 기상과 대기 데이터 병합
        combined = pd.merge(weather_df, air_df, on="date", how="inner")
        return combined.sort_values("date", ascending=False)
    else:
        st.error("대기오염 데이터 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

# 데이터 로드
df_env = get_latest_environment_data()

if not df_env.empty:
    # 가장 최신 데이터 (기준일자)
    latest_row = df_env.iloc[0]
    base_date = latest_row['date'].strftime('%Y년 %m월 %d일')

    # ============================================================
    # 2. 대시보드 - 현재 환경 현황
    # ============================================================
    st.subheader(f"📅 기준 일자: {base_date} (영천 관측소)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ 평균 기온", f"{latest_row['temp']} °C")
    m2.metric("💧 평균 습도", f"{latest_row['humidity']} %")
    m3.metric("☔ 강수량", f"{latest_row['rainfall']} mm")
    m4.metric("💨 평균 풍속", f"{latest_row['wind']} m/s")

    a1, a2, a3, a4, a5, a6 = st.columns(6)
    a1.metric("🌫️ PM10", f"{latest_row['pm10']}")
    a2.metric("😷 PM2.5", f"{latest_row['pm25']}")
    a3.metric("☁️ CO", f"{latest_row['co']}")
    a4.metric("☀️ O3", f"{latest_row['o3']}")
    a5.metric("🚗 NO2", f"{latest_row['no2']}")
    a6.metric("🏭 SO2", f"{latest_row['so2']}")

    # ============================================================
    # 3. 모델 학습 (파생변수 포함)
    # ============================================================
    def train_model():
        # 파생변수 생성 (학습용)
        train_df = df_env.copy()
        a, b = 17.27, 237.7
        gamma = (a * train_df["temp"] / (b + train_df["temp"])) + np.log(train_df["humidity"].clip(lower=1) / 100)
        train_df["dew_point"] = (b * gamma) / (a - gamma)
        train_df["dew_gap"] = train_df["temp"] - train_df["dew_point"]
        
        # 재질별 확장 시뮬레이션 데이터 구축 (Random Forest용)
        # (현업에서는 실제 훼손 이력 데이터를 사용하나, 여기서는 로직 기반 학습)
        mats = {'목조':0, '석조':1, '금속':2, '벽화':3}
        exps = {'실외':0, '실내':1}
        
        full_rows = []
        for _, r in train_df.iterrows():
            for m_name, m_code in mats.items():
                for e_name, e_code in exps.items():
                    row = r.to_dict()
                    row.update({'mat_code': m_code, 'exp_code': e_code})
                    
                    # 라벨링 로직
                    score = 0
                    adj = 0.4 if e_name == '실내' else 1.0
                    if m_name == '목조' and (r['humidity'] > 85 or r['humidity'] < 25): score += 2.5 * adj
                    if m_name == '석조' and r['rainfall'] > 15 and r['so2'] > 0.005: score += 3.0 * adj
                    if r['o3'] > 0.08 or r['dew_gap'] < 2: score += 2.0 * adj
                    
                    row['danger_level'] = 2 if score >= 2.5 else (1 if score >= 1.2 else 0)
                    full_rows.append(row)
        
        data = pd.DataFrame(full_rows)
        features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'co', 'o3', 'no2', 'so2', 'mat_code', 'exp_code', 'dew_gap']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(data[features].fillna(0), data['danger_level'])
        return model

    model = train_model()

    # ============================================================
    # 4. 분석 설정 및 위험도 예측
    # ============================================================
    st.divider()
    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.subheader("🔍 분석 대상 설정")
        mat = st.selectbox("문화재 재질", ["목조", "석조", "금속", "벽화"])
        exp = st.radio("노출 형태", ["실외", "실내"], horizontal=True)
        
        # 선택된 재질의 주요 위험 요인 설명
        risk_info = {
            "목조": "고습도(부패) 및 저습도(균열), 결로에 취약",
            "석조": "산성비(SO2) 및 동결융해, 미세먼지 풍화에 취약",
            "금속": "습도와 대기오염 물질에 의한 부식 및 산화에 취약",
            "벽화": "오존(퇴색) 및 온습도 변동에 의한 박락에 취약"
        }
        st.info(f"**{mat} 특징:** {risk_info[mat]}")

    with c2:
        st.subheader("🤖 AI 위험도 판정")
        
        # 입력 데이터 구성
        a, b = 17.27, 237.7
        current_gamma = (a * latest_row['temp'] / (b + latest_row['temp'])) + np.log(latest_row['humidity'] / 100)
        current_dew_gap = latest_row['temp'] - ((b * current_gamma) / (a - current_gamma))
        
        input_data = pd.DataFrame([{
            'temp': latest_row['temp'], 'humidity': latest_row['humidity'], 
            'rainfall': latest_row['rainfall'], 'wind': latest_row['wind'],
            'pm10': latest_row['pm10'], 'pm25': latest_row['pm25'], 
            'co': latest_row['co'], 'o3': latest_row['o3'], 
            'no2': latest_row['no2'], 'so2': latest_row['so2'],
            'mat_code': {'목조':0, '석조':1, '금속':2, '벽화':3}[mat],
            'exp_code': {'실외':0, '실내':1}[exp],
            'dew_gap': current_dew_gap
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        # 시각적 결과 표시
        status = ["안전 (Safe)", "주의 (Caution)", "위험 (Danger)"]
        colors = ["#2E7D32", "#F9A825", "#C62828"]
        
        st.markdown(f"""
            <div style="background-color:{colors[pred]}; padding:30px; border-radius:15px; text-align:center;">
                <h2 style="color:white; margin:0;">{status[pred]}</h2>
                <p style="color:white; font-size:1.2rem; margin-top:10px;">신뢰도: {prob[pred]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 위험 등급별 확률 그래프
        prob_df = pd.DataFrame({'등급': ["안전", "주의", "위험"], '확률': prob})
        st.bar_chart(data=prob_df, x='등급', y='확률')

    # ============================================================
    # 5. 관리 가이드라인
    # ============================================================
    st.divider()
    st.subheader("📋 향후 관리 제언")
    
    advice_col1, advice_col2 = st.columns(2)
    with advice_col1:
        if pred == 2:
            st.error(f"🚨 **긴급 점검 필요**: 현재 {mat} 문화재가 훼손될 가능성이 매우 높습니다. 외부 노출을 최소화하고 보존 환경을 점검하십시오.")
        elif pred == 1:
            st.warning(f"⚠️ **주의 관찰**: 환경 요인이 불안정합니다. {mat} 표면의 상태 변화를 정기적으로 모니터링하십시오.")
        else:
            st.success(f"✅ **상태 양호**: 현재 {mat} 문화재 보존에 적합한 환경이 유지되고 있습니다.")

    with advice_col2:
        st.markdown(f"""
        **주요 분석 근거:**
        - **재질**: {mat} / **노출**: {exp}
        - **결로 지수**: {'위험' if current_dew_gap < 3 else '안전'} (차이: {current_dew_gap:.2f})
        - **오염 부하**: {'높음' if latest_row['pm10'] > 80 else '보통'}
        """)

else:
    st.warning("데이터를 불러오는 중입니다. 잠시만 기다려주세요...")
