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
    weather_path = "data/processed/[2016_2025] yeongcheon_weather_daily.csv"
    air_path = "data/processed/[2019_2025] air_quality.csv"
    
    if not os.path.exists(weather_path) or not os.path.exists(air_path):
        return None, None

    # 데이터 병합 및 전처리
    w_df = pd.read_csv(weather_path)
    a_df = pd.read_csv(air_path)
    w_df = w_df.rename(columns={
        'avg_temperature_c': 'temp', 'daily_precipitation_mm': 'rainfall',
        'avg_wind_speed_ms': 'wind', 'avg_relative_humidity_pct': 'humidity'
    })
    m_df = pd.merge(w_df, a_df, on='date', how='inner')

    # 재질/노출별 학습 데이터 생성
    mats = {'목조':0, '석조':1, '금속':2, '벽화':3, '기타':4}
    exps = {'실외':0, '실내':1}
    
    train_rows = []
    for _, r in m_df.tail(1200).iterrows(): # 약 3년치 데이터 학습
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                score = 0
                adj = 0.4 if e_name == '실내' else 1.0
                if m_name == '목조' and (r['humidity'] > 80 or r['humidity'] < 30): score += 2.5 * adj
                if m_name == '석조' and r['rainfall'] > 15: score += 2.0 * adj
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

ai_model, feature_names = train_heritage_model()

# ==========================================================
# 2. 실시간 환경 데이터 수집 (API)
# ==========================================================
# [기상 데이터]
tm, temp, humidity, rainfall, wind_speed = "-", "-", "-", "-", "-"
try:
    asos_params = {"serviceKey": SERVICE_KEY, "numOfRows": "1", "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR", "startDt": base_date, "startHh": base_hour, "endDt": base_date, "endHh": base_hour, "stnIds": "281"}
    res = requests.get("https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList", params=asos_params, timeout=10).json()
    item = res["response"]["body"]["items"]["item"][0]
    tm, temp, humidity, rainfall, wind_speed = item["tm"], item["ta"], item["hm"], item["rn"], item["ws"]
except: pass

# [대기 데이터]
pm10, pm25, o3, no2, co, so2, data_time = "-", "-", "-", "-", "-", "-", "-"
try:
    air_params = {"serviceKey": SERVICE_KEY, "returnType": "json", "numOfRows": "100", "sidoName": "경북", "ver": "1.0"}
    res = requests.get("https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty", params=air_params, timeout=10).json()
    target = next((i for i in res["response"]["body"]["items"] if "영천" in i["stationName"]), None)
    if target:
        pm10, pm25, o3, no2, co, so2, data_time = target["pm10Value"], target["pm25Value"], target["o3Value"], target["no2Value"], target["coValue"], target["so2Value"], target["dataTime"]
except: pass

# ==========================================================
# 3. UI 디자인 요소 (CSS)
# ==========================================================
card_style = "background-color:#f8f9fa; padding:22px; border-radius:20px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.05); height:350px;"
title_style = "font-size:24px; font-weight:700; margin-bottom:14px; color:#1f2937;"
label_style = "font-size:14px; color:#6b7280; margin-bottom:4px;"
value_style = "font-size:22px; font-weight:700; color:#111827; margin-bottom:18px;"
time_style = "font-size:13px; color:#9ca3af; margin-top:12px; position:absolute; bottom:20px;"

# ==========================================================
# 4. 상단 대시보드 출력
# ==========================================================
st.markdown("<h1 style='font-size:30px;'>🏛 공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측</h1>", unsafe_allow_html=True)
st.markdown("영천 지역 문화재와 공공 환경데이터를 분석하여 훼손 위험을 사전에 예측합니다.")
st.divider()

st.markdown('<h3 style="font-size:25px; margin-bottom:10px;">🌿 영천시 환경 데이터 및 문화재 현황</h3>', unsafe_allow_html=True)

left, center, right = st.columns([1.4, 2.0, 1.0])

# 1열 기상
with left:
    st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🌦 기상 환경</div><hr><div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px;">
    <div><div style="{label_style}">🌡 기온</div><div style="{value_style}">{temp} °C</div></div>
    <div><div style="{label_style}">💧 습도</div><div style="{value_style}">{humidity} %</div></div>
    <div><div style="{label_style}">🌧 강수량</div><div style="{value_style}">{rainfall} mm</div></div>
    <div><div style="{label_style}">💨 풍속</div><div style="{value_style}">{wind_speed} m/s</div></div>
    </div><div style="{time_style}">⏱ 측정 시각 : {tm}</div></div>""", unsafe_allow_html=True)

# 2열 대기
with center:
    st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🌫 대기오염 현황</div><hr><div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-top:20px;">
    <div><div style="{label_style}">PM10</div><div style="{value_style}">{pm10}</div><div style="{label_style}">O₃</div><div style="{value_style}">{o3}</div></div>
    <div><div style="{label_style}">PM2.5</div><div style="{value_style}">{pm25}</div><div style="{label_style}">NO₂</div><div style="{value_style}">{no2}</div></div>
    <div><div style="{label_style}">CO</div><div style="{value_style}">{co}</div><div style="{label_style}">SO₂</div><div style="{value_style}">{so2}</div></div>
    </div><div style="{time_style}">⏱ 측정 시각 : {data_time}</div></div>""", unsafe_allow_html=True)

# 3열 문화재 요약
with right:
    heritage_feature = pd.read_csv("data/processed/yc_heritage_feature.csv")
    st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🏛 문화재 현황</div><hr><div style="margin-top:20px;">
    <div style="{label_style}">분석 문화재 수</div><div style="{value_style}">{len(heritage_feature)}개</div><br>
    <div style="{label_style}">📍 분석 지역</div><div style="{value_style}">경북 영천시</div></div></div>""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# 5. AI 위험도 판별 결과 (테이블)
# ==========================================================
st.subheader("🎯 개별 문화재 실시간 위험도 분석 결과")

def safe_cast(val):
    try: return float(val)
    except: return 0.0

if ai_model:
    # 현재 환경 데이터를 모델 입력용으로 변환
    curr_env = {
        'temp': safe_cast(temp), 'humidity': safe_cast(humidity), 'rainfall': safe_cast(rainfall),
        'wind': safe_cast(wind_speed), 'pm10': safe_cast(pm10), 'pm25': safe_cast(pm25),
        'so2': safe_cast(so2), 'no2': safe_cast(no2), 'co': safe_cast(co), 'o3': safe_cast(o3)
    }
    
    mat_map = {'목조':0, '석조':1, '금속':2, '벽화':3}
    exp_map = {'실외':0, '실내':1}
    
    analysis_results = []
    for _, row in heritage_feature.iterrows():
        m_code = mat_map.get(row['재질'], 4)
        e_code = exp_map.get(row['노출형태'], 0)
        
        input_vec = pd.DataFrame([{**curr_env, 'mat_code': m_code, 'exp_code': e_code}])
        prediction = ai_model.predict(input_vec[feature_names])[0]
        probability = ai_model.predict_proba(input_vec[feature_names])[0]
        
        analysis_results.append({
            '문화재명': row['문화재명(국문)'],
            '재질': row['재질'],
            '노출': row['노출형태'],
            '위험수치': round(probability[2] * 100, 1),
            '등급': ['✅ 안전', '⚠️ 주의', '🚨 위험'][prediction]
        })
    
    res_df = pd.DataFrame(analysis_results)
    
    # 등급별 요약 메트릭
    c1, c2, c3 = st.columns(3)
    c1.metric("안전 등급", f"{len(res_df[res_df['등급']=='✅ 안전'])}개")
    c2.metric("주의 등급", f"{len(res_df[res_df['등급']=='⚠️ 주의'])}개")
    c3.metric("위험 등급", f"{len(res_df[res_df['등급']=='🚨 위험'])}개")
    
    # 분석 테이블 출력
    st.dataframe(
        res_df.sort_values('위험수치', ascending=False),
        column_config={
            "위험수치": st.column_config.ProgressColumn("훼손 위험 지수", min_value=0, max_value=100, format="%f%%"),
            "등급": st.column_config.TextColumn("AI 판정")
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.error("분석을 위한 과거 데이터 파일을 찾을 수 없습니다.")

st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
