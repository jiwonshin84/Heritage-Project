import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# 0. 페이지 설정
# ============================================================
st.set_page_config(page_title="영천 문화재 훼손 위험 예측", layout="wide")

st.title("🏛️ 영천 지역 문화재 훼손 위험 분석 시스템")
st.markdown("기상 및 대기오염 통합 데이터를 분석하여 재질별 위험 등급을 예측합니다.")

# ============================================================
# 1. 데이터 로드 및 통합 전처리
# ============================================================
@st.cache_data
def load_and_process_data():
    # 파일 경로 설정
    weather_path = "data/processed/[2016_2025] yeongcheon_weather_daily.csv"
    air_path = "data/processed/[2019_2025] air_quality.csv"
    
    if not os.path.exists(weather_path) or not os.path.exists(air_path):
        st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return pd.DataFrame()

    # 1) 기상 데이터 로드 (제시해주신 컬럼명 반영)
    # 컬럼: date, avg_temperature_c, daily_precipitation_mm, avg_wind_speed_ms, avg_relative_humidity_pct
    weather = pd.read_csv(weather_path)
    weather['date'] = pd.to_datetime(weather['date'])
    weather = weather.rename(columns={
        'avg_temperature_c': 'temp',
        'daily_precipitation_mm': 'rainfall',
        'avg_wind_speed_ms': 'wind',
        'avg_relative_humidity_pct': 'humidity'
    })

    # 2) 대기질 데이터 로드
    air = pd.read_csv(air_path)
    air['date'] = pd.to_datetime(air['date'])

    # 3) 데이터 병합
    df = pd.merge(weather, air, on='date', how='inner').sort_values('date')

    # 4) 파생 변수 생성 (사용자 로직 반영)
    # Magnus 공식 이슬점 계산
    a, b = 17.27, 237.7
    h_clip = df["humidity"].clip(lower=1)
    gamma = (a * df["temp"] / (b + df["temp"])) + np.log(h_clip / 100)
    df["dew_point"] = (b * gamma) / (a - gamma)
    df["dew_gap"] = df["temp"] - df["dew_point"]

    # 변화율 및 이동평균
    df["temp_change"] = df["temp"].diff().fillna(0)
    df["humi_change"] = df["humidity"].diff().fillna(0)
    df["shock_risk"] = abs(df["temp_change"]) + abs(df["humi_change"])
    df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3).mean().fillna(df["pm10"] + df["pm25"])
    
    return df

df_base = load_and_process_data()

# ============================================================
# 2. 모델 학습 (재질/노출 형태 시뮬레이션)
# ============================================================
def train_heritage_model(base_df):
    mats = {'목조':0, '석조':1, '금속':2, '벽화':3}
    exps = {'실외':0, '실내':1}
    
    full_data = []
    # 데이터가 너무 많을 경우 최근 데이터 위주로 학습 (성능 최적화)
    sample_df = base_df.tail(1000) 

    for _, r in sample_df.iterrows():
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                row = r.to_dict()
                row.update({'mat_code': m_code, 'exp_code': e_code})
                
                # 라벨링 로직 (위험도 점수제)
                score = 0
                adj = 0.4 if e_name == '실내' else 1.0
                
                if m_name == '목조':
                    if r['shock_risk'] > 8: score += 2 * adj
                    if r['humidity'] > 80: score += 1.5 * adj
                elif m_name == '석조':
                    if r['rainfall'] > 15 and r['so2'] > 0.005: score += 3 * adj
                elif m_name == '벽화':
                    if r['o3'] > 0.06 or r['dew_gap'] < 2: score += 2.5 * adj
                
                row['danger_level'] = 2 if score >= 2.5 else (1 if score >= 1.2 else 0)
                full_data.append(row)

    train_df = pd.DataFrame(full_data)
    features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'co', 'o3', 'no2', 'so2', 
                'mat_code', 'exp_code', 'dew_gap', 'shock_risk', 'pm_load']
    
    X = train_df[features].fillna(0)
    y = train_df['danger_level']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, features

# 데이터 로드 확인 후 실행
if not df_base.empty:
    model, feature_list = train_heritage_model(df_base)
    latest_data = df_base.iloc[-1] # 데이터셋의 가장 마지막 날짜

    # ============================================================
    # 3. 대시보드 UI
    # ============================================================
    st.subheader(f"📅 데이터 분석 기준일: {latest_data['date'].strftime('%Y-%m-%d')}")
    
    # 상단 지표
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡️ 평균 기온", f"{latest_data['temp']} °C")
    c2.metric("💧 평균 습도", f"{latest_row['humidity'] if 'latest_row' in locals() else latest_data['humidity']} %")
    c3.metric("☔ 강수량", f"{latest_data['rainfall']} mm")
    c4.metric("🌫️ 미세먼지(PM10)", f"{latest_data['pm10']}")

    st.divider()

    col_input, col_result = st.columns([1, 1.5])

    with col_input:
        st.write("### 🔍 분석 조건 선택")
        selected_mat = st.selectbox("문화재 재질", ["목조", "석조", "금속", "벽화"])
        selected_exp = st.radio("노출 형태", ["실외", "실내"], horizontal=True)
        
        st.write("---")
        st.write("**현재 환경 상세 수치**")
        # 최근 데이터를 기반으로 하되 사용자가 조정 가능하게 설정
        t = st.slider("기온 수정", -20.0, 40.0, float(latest_data['temp']))
        h = st.slider("습도 수정", 0, 100, int(latest_data['humidity']))
        
    with col_result:
        st.write("### 🤖 AI 위험도 판정")
        
        # 입력값에 따른 파생변수 즉석 계산
        m_code = {'목조':0, '석조':1, '금속':2, '벽화':3}[selected_mat]
        e_code = {'실외':0, '실내':1}[selected_exp]
        
        # 간이 이슬점 계산
        current_gamma = (17.27 * t / (237.7 + t)) + np.log(max(h, 1) / 100)
        current_dew_point = (237.7 * current_gamma) / (17.27 - current_gamma)
        current_dew_gap = t - current_dew_point

        input_row = pd.DataFrame([{
            'temp': t, 'humidity': h, 'rainfall': latest_data['rainfall'], 'wind': latest_data['wind'],
            'pm10': latest_data['pm10'], 'pm25': latest_data['pm25'], 
            'co': latest_data['co'], 'o3': latest_data['o3'], 
            'no2': latest_data['no2'], 'so2': latest_data['so2'],
            'mat_code': m_code, 'exp_code': e_code,
            'dew_gap': current_dew_gap, 
            'shock_risk': latest_data['shock_risk'], 
            'pm_load': latest_data['pm_load']
        }])

        pred = model.predict(input_row[feature_list])[0]
        prob = model.predict_proba(input_row[feature_list])[0]

        # 결과 띠(Banner) 출력
        res_label = ["안전 (Safe)", "주의 (Caution)", "위험 (Danger)"]
        res_color = ["#2E7D32", "#F9A825", "#C62828"]
        
        st.markdown(f"""
            <div style="background-color:{res_color[pred]}; padding:25px; border-radius:15px; text-align:center; color:white;">
                <h2 style="margin:0;">{res_label[pred]}</h2>
                <p style="font-size:1.1rem; opacity:0.9;">분석 신뢰도: {prob[pred]*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # 등급별 확률 그래프
        st.write("")
        chart_data = pd.DataFrame({'등급': ["안전", "주의", "위험"], '확률': prob})
        st.bar_chart(data=chart_data, x='등급', y='확률')

    st.divider()
    # 재질별 맞춤 조언
    st.write("### 📋 관리 가이드라인")
    advices = {
        "목조": "결로와 습도 급변에 의한 갈라짐을 주의하십시오. 실내 환경일 경우 항온항습기 점검이 필요합니다.",
        "석조": "산성비 및 대기 오염에 의한 표면 박락을 주의하십시오. 고위험 시 보호각 설치를 검토하십시오.",
        "금속": "산화 부식 방지를 위해 습도 60% 이하 유지를 권장합니다.",
        "벽화": "안료 퇴색 방지를 위해 자외선 및 오존 농도 차단에 주력하십시오."
    }
    st.info(f"**{selected_mat} 보존 팁:** {advices[selected_mat]}")

else:
    st.error("데이터 로드에 실패했습니다. 파일명과 경로를 다시 확인해주세요.")
