import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# 0. 설정 및 API 키
# ==========================================================
st.set_page_config(
    page_title="영천 지역 문화재 훼손 위험 예측",
    page_icon="🏛",
    layout="wide"
)

SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
now = datetime.now(ZoneInfo("Asia/Seoul"))
yesterday = now - timedelta(days=1)
base_date = yesterday.strftime("%Y%m%d")
base_hour = "23"

# ==========================================================
# 1. 모델 학습 (과거 데이터를 기반으로 위험 패턴 학습)
# ==========================================================
@st.cache_resource
def train_heritage_model():
    # 데이터 경로
    weather_path = "data/processed/[2016_2025] yeongcheon_weather_daily.csv"
    air_path = "data/processed/[2019_2025] air_quality.csv"
    
    if not os.path.exists(weather_path) or not os.path.exists(air_path):
        return None, None

    # 기상/대기 데이터 로드 및 병합
    w_df = pd.read_csv(weather_path)
    a_df = pd.read_csv(air_path)
    w_df = w_df.rename(columns={
        'avg_temperature_c': 'temp', 'daily_precipitation_mm': 'rainfall',
        'avg_wind_speed_ms': 'wind', 'avg_relative_humidity_pct': 'humidity'
    })
    m_df = pd.merge(w_df, a_df, on='date', how='inner')

    # 파생변수 및 라벨링 시뮬레이션
    mats = {'목조':0, '석조':1, '금속':2, '벽화':3, '기타':4}
    exps = {'실외':0, '실내':1}
    
    train_rows = []
    # 최근 1000일 데이터를 사용하여 재질/노출별 위험 상황 학습
    for _, r in m_df.tail(1000).iterrows():
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                score = 0
                adj = 0.4 if e_name == '실내' else 1.0
                # 위험 점수 로직
                if m_name == '목조' and (r['humidity'] > 80 or r['humidity'] < 30): score += 2.5 * adj
                if m_name == '석조' and r['rainfall'] > 10: score += 2.0 * adj
                if r['pm10'] > 80 or r['o3'] > 0.08: score += 1.5 * adj
                
                danger = 2 if score >= 2.5 else (1 if score >= 1.2 else 0)
                train_rows.append({
                    'temp': r['temp'], 'humidity': r['humidity'], 'rainfall': r['rainfall'],
                    'wind': r['wind'], 'pm10': r['pm10'], 'pm25': r['pm25'],
                    'so2': r['so2'], 'no2': r['no2'], 'co': r['co'], 'o3': r['o3'],
                    'mat_code': m_code, 'exp_code': e_code, 'target': danger
                })
    
    tdf = pd.DataFrame(train_rows)
    features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'so2', 'no2', 'co', 'o3', 'mat_code', 'exp_code']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(tdf[features].fillna(0), tdf['target'])
    return model, features

# 모델 로드
ai_model, feature_names = train_heritage_model()

# ==========================================================
# 2. 실시간 환경 데이터 수집 (API)
# ==========================================================
# [ASOS 기상 데이터]
tm, temp, humidity, rainfall, wind_speed = "-", "-", "-", "-", "-"
try:
    asos_params = {
        "serviceKey": SERVICE_KEY, "pageNo": "1", "numOfRows": "1", "dataType": "JSON",
        "dataCd": "ASOS", "dateCd": "HR", "startDt": base_date, "startHh": base_hour,
        "endDt": base_date, "endHh": base_hour, "stnIds": "281"
    }
    res = requests.get("https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList", params=asos_params, timeout=10)
    item = res.json()["response"]["body"]["items"]["item"][0]
    tm, temp, humidity, rainfall, wind_speed = item["tm"], item["ta"], item["hm"], item["rn"], item["ws"]
except: pass

# [대기오염 데이터]
pm10, pm25, o3, no2, co, so2, data_time = "-", "-", "-", "-", "-", "-", "-"
try:
    air_params = {"serviceKey": SERVICE_KEY, "returnType": "json", "numOfRows": "100", "sidoName": "경북", "ver": "1.0"}
    res = requests.get("https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty", params=air_params, timeout=10)
    items = res.json()["response"]["body"]["items"]
    target = next((i for i in items if "영천" in i["stationName"]), None)
    if target:
        pm10, pm25, o3, no2, co, so2, data_time = target["pm10Value"], target["pm25Value"], target["o3Value"], target["no2Value"], target["coValue"], target["so2Value"], target["dataTime"]
except: pass

# ==========================================================
# 3. 메인 UI 및 대시보드
# ==========================================================
st.markdown("<h1 style='font-size:30px;'>🏛 영천 지역 문화재 훼손 위험 실시간 분석</h1>", unsafe_allow_html=True)
st.divider()

# 상단 대시보드 (기존 스타일 유지)
l, c, r = st.columns([1.4, 2.0, 1.0])
card_style = "background-color:#f8f9fa; padding:20px; border-radius:15px; border:1px solid #e5e7eb; height:320px;"

with l:
    st.markdown(f"<div style='{card_style}'><h4>🌦 기상 환경</h4><hr><b>🌡 기온:</b> {temp} °C<br><b>💧 습도:</b> {humidity} %<br><b>🌧 강수:</b> {rainfall} mm<br><b>💨 풍속:</b> {wind_speed} m/s<p style='font-size:12px; color:gray; margin-top:20px;'>⏱ {tm}</p></div>", unsafe_allow_html=True)
with c:
    st.markdown(f"<div style='{card_style}'><h4>🌫 대기오염 현황</h4><hr><div style='display:grid; grid-template-columns:1fr 1fr; gap:10px;'><div><b>PM10:</b> {pm10}<br><b>PM2.5:</b> {pm25}<br><b>O3:</b> {o3}</div><div><b>NO2:</b> {no2}<br><b>CO:</b> {co}<br><b>SO2:</b> {so2}</div></div><p style='font-size:12px; color:gray; margin-top:50px;'>⏱ {data_time}</p></div>", unsafe_allow_html=True)
with r:
    heritage_feature = pd.read_csv("data/processed/yc_heritage_feature.csv")
    st.markdown(f"<div style='{card_style}'><h4>🏛 문화재 현황</h4><hr><b>분석 대상:</b> {len(heritage_feature)}개<br><b>지역:</b> 경북 영천시</div>", unsafe_allow_html=True)

st.divider()

# ==========================================================
# 4. 실시간 위험도 판별 실행
# ==========================================================
st.subheader("🎯 개별 문화재별 현재 위험도 판별")

# 수치 데이터 전처리 (에러 방지)
def safe_float(v):
    try: return float(v)
    except: return 0.0

if ai_model is not None:
    curr_env = {
        'temp': safe_float(temp), 'humidity': safe_float(humidity), 'rainfall': safe_float(rainfall),
        'wind': safe_float(wind_speed), 'pm10': safe_float(pm10), 'pm25': safe_float(pm25),
        'so2': safe_float(so2), 'no2': safe_float(no2), 'co': safe_float(co), 'o3': safe_float(o3)
    }
    
    mat_map = {'목조':0, '석조':1, '금속':2, '벽화':3}
    exp_map = {'실외':0, '실내':1}
    
    results = []
    for _, h in heritage_feature.iterrows():
        m_code = mat_map.get(h['재질'], 4)
        e_code = exp_map.get(h['노출형태'], 0)
        
        input_data = pd.DataFrame([{**curr_env, 'mat_code': m_code, 'exp_code': e_code}])
        pred = ai_model.predict(input_data[feature_names])[0]
        prob = ai_model.predict_proba(input_data[feature_names])[0]
        
        results.append({
            '문화재명': h['문화재명(국문)'], '재질': h['재질'], '노출': h['노출형태'],
            '위험수치': round(prob[2] * 100, 1), '등급': ['안전', '주의', '위험'][pred]
        })
    
    res_df = pd.DataFrame(results)
    
    # 판정 통계
    s1, s2, s3 = st.columns(3)
    s1.success(f"✅ 안전: {len(res_df[res_df['등급']=='안전'])}개")
    s2.warning(f"⚠️ 주의: {len(res_df[res_df['등급']=='주의'])}개")
    s3.error(f"🚨 위험: {len(res_df[res_df['등급']=='위험'])}개")
    
    # 결과 테이블
    st.dataframe(
        res_df.sort_values('위험수치', ascending=False),
        column_config={
            "위험수치": st.column_config.ProgressColumn("훼손 위험 지수", min_value=0, max_value=100, format="%f%%"),
            "등급": st.column_config.TextColumn("판정 결과")
        },
        use_container_width=True, hide_index=True
    )
else:
    st.error("학습 데이터 파일을 찾을 수 없어 AI 분석을 시작할 수 없습니다.")

st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
