import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# ============================================================
# 1. 페이지 설정 및 제목
# ============================================================
st.set_page_config(page_title="문화재 훼손 위험 예측 시스템", layout="wide")
st.title("🚨 실시간 기반 문화재 훼손 위험 예측")
st.markdown("최신 기상 및 대기오염 데이터를 분석하여 재질별 위험 등급을 산출합니다.")

# ============================================================
# 2. 데이터 로드 및 전처리 함수
# ============================================================
@st.cache_data
def load_and_merge_data():
    # 1) 기상 데이터 수집 (영천 관측소 281, 최근 데이터 위주)
    ASOS_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
    current_year = datetime.now().year
    
    def fetch_weather(year):
        url = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
        params = {
            "serviceKey": ASOS_SERVICE_KEY, "numOfRows": "400", "dataType": "JSON",
            "dataCd": "ASOS", "dateCd": "DAY", "startDt": f"{year}0101", 
            "endDt": f"{year}1231", "stnIds": "281"
        }
        try:
            res = requests.get(url, params=params).json()
            return pd.DataFrame(res["response"]["body"]["items"]["item"])
        except: return pd.DataFrame()

    # 최근 3개년 데이터 수집 (예측 모델 학습용)
    weather_list = [fetch_weather(y) for y in range(current_year-2, current_year+1)]
    weather = pd.concat(weather_list, ignore_index=True)
    
    # 기상 컬럼 정제
    weather = weather[["tm", "avgTa", "avgRhm", "sumRn", "avgWs"]].copy()
    weather.columns = ["date", "temp", "humidity", "rainfall", "wind"]
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["temp", "humidity", "rainfall", "wind"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather["rainfall"] = weather["rainfall"].fillna(0)

    # 2) 대기오염 데이터 로드 (로컬 경로)
    air_path = "data/processed/[2019_2025] air_quality.csv"
    if os.path.exists(air_path):
        air = pd.read_csv(air_path)
        air["date"] = pd.to_datetime(air["date"])
    else:
        # 파일이 없을 경우 예시 데이터 생성 (데모용)
        st.error(f"데이터 파일을 찾을 수 없습니다: {air_path}")
        return None

    # 병합
    df = pd.merge(weather, air, on="date", how="inner").sort_values("date")
    return df

# ============================================================
# 3. 파생 변수 및 모델 학습 로직
# ============================================================
def train_model(df_base):
    # 파생 변수 생성
    df = df_base.copy()
    a, b = 17.27, 237.7
    gamma = (a * df["temp"] / (b + df["temp"])) + np.log(df["humidity"].clip(lower=1) / 100)
    df["dew_point"] = (b * gamma) / (a - gamma)
    df["dew_gap"] = df["temp"] - df["dew_point"]
    df["shock_risk"] = (abs(df["temp"].diff()) + abs(df["humidity"].diff())).fillna(0)
    df["pm_load"] = (df["pm10"] + df["pm25"]).rolling(3).mean().fillna(0)
    
    # 재질 데이터 확장 (시뮬레이션)
    mats = {'목조':0, '석조':1, '금속':2, '벽화':3}
    exps = {'실외':0, '실내':1}
    
    expanded_rows = []
    for _, row in df.iterrows():
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                r = row.to_dict()
                r.update({'mat_code': m_code, 'exp_code': e_code})
                
                # 라벨링 로직 (간소화)
                score = 0
                adj = 0.4 if e_name == '실내' else 1.0
                if m_name == '목조' and r['shock_risk'] > 10: score += 2 * adj
                if m_name == '석조' and r['rainfall'] > 10 and r['so2'] > 0.005: score += 3 * adj
                if r['dew_gap'] < 2: score += 2 * adj
                
                r['danger_level'] = 2 if score >= 3 else (1 if score >= 1.5 else 0)
                expanded_rows.append(r)
                
    full_df = pd.DataFrame(expanded_rows)
    features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'so2', 'no2', 'o3', 
                'mat_code', 'exp_code', 'shock_risk', 'dew_gap', 'pm_load']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(full_df[features].fillna(0), full_df['danger_level'])
    return model

# 데이터 로드 및 모델 준비
df_raw = load_and_merge_data()
if df_raw is not None:
    model = train_model(df_raw)
    last_data = df_raw.iloc[-1] # 가장 최근 관측치

    # ============================================================
    # 4. UI 레이아웃
    # ============================================================
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🛠 조건 설정")
        selected_mat = st.selectbox("문화재 재질", ["목조", "석조", "금속", "벽화"])
        selected_exp = st.radio("노출 형태", ["실외", "실내"])
        
        st.info(f"📅 기준 일자: {last_data['date'].strftime('%Y-%m-%d')}")
        
        # 실시간 수치 조정 (기본값은 최근 관측치)
        st.write("**현재 환경 수치 (수정 가능)**")
        t = st.slider("기온 (°C)", -20.0, 40.0, float(last_data['temp']))
        h = st.slider("습도 (%)", 0, 100, int(last_data['humidity']))
        p10 = st.number_input("미세먼지(PM10)", 0, 500, int(last_data['pm10']))
        so2 = st.number_input("이산화황(SO2)", 0.0, 0.1, float(last_data['so2']), format="%.3f")

    with col2:
        st.subheader("🔮 예측 결과")
        
        # 예측용 파생변수 계산
        m_code = {"목조":0, "석조":1, "금속":2, "벽화":3}[selected_mat]
        e_code = {"실외":0, "실내":1}[selected_exp]
        
        # 간이 파생변수 생성
        input_data = pd.DataFrame([{
            'temp': t, 'humidity': h, 'rainfall': last_data['rainfall'], 'wind': last_data['wind'],
            'pm10': p10, 'pm25': last_data['pm25'], 'so2': so2, 'no2': last_data['no2'], 'o3': last_data['o3'],
            'mat_code': m_code, 'exp_code': e_code,
            'shock_risk': 0, 'dew_gap': t - (t - ((100-h)/5)), 'pm_load': p10 + last_data['pm25']
        }])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        # 결과 카드 출력
        colors = ["#28a745", "#ffc107", "#dc3545"]
        labels = ["안전 (Safe)", "주의 (Caution)", "위험 (Danger)"]
        
        st.markdown(f"""
            <div style="background-color:{colors[pred]}; padding:20px; border-radius:10px; text-align:center;">
                <h1 style="color:white; margin:0;">{labels[pred]}</h1>
                <p style="color:white; font-size:1.2em;">위험 확률: {prob[pred]*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # 상세 지표 시각화
        st.write("### 📊 등급별 판단 확률")
        prob_df = pd.DataFrame({
            '상태': ['안전', '주의', '위험'],
            '확률': prob
        })
        st.bar_chart(data=prob_df, x='상태', y='확률')

    st.divider()
    st.subheader("💡 관리 권고 사항")
    if pred == 2:
        st.error(f"**[{selected_mat}]** 문화재의 부식 및 훼손이 우려되는 환경입니다. 즉시 현장 점검 및 보호 덮개 설치를 권고합니다.")
    elif pred == 1:
        st.warning("환경 변화를 주시하십시오. 특히 결로 및 표면 오염에 대한 정기적인 모니터링이 필요합니다.")
    else:
        st.success("현재 문화재를 보존하기에 양호한 환경입니다.")
