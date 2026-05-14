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

# 전역 변수 역할을 할 세션 스테이트 초기화 (위험 문화재 수)
if 'danger_count' not in st.session_state:
    st.session_state['danger_count'] = 0

# ==========================================================
# 1. 모델 학습 로직
# ==========================================================
@st.cache_resource
def train_heritage_model():
    weather_path = "data/processed/[2016_2025] yeongcheon_weather_daily.csv"
    air_path = "data/processed/[2019_2025] air_quality.csv"
    
    if not os.path.exists(weather_path) or not os.path.exists(air_path):
        return None, None

    w_df = pd.read_csv(weather_path)
    a_df = pd.read_csv(air_path)
    w_df = w_df.rename(columns={
        'avg_temperature_c': 'temp', 'daily_precipitation_mm': 'rainfall',
        'avg_wind_speed_ms': 'wind', 'avg_relative_humidity_pct': 'humidity'
    })
    m_df = pd.merge(w_df, a_df, on='date', how='inner')

    mats = {'목조':0, '석조':1, '금속':2, '벽화':3, '기타':4}
    exps = {'실외':0, '실내':1, '반실외':2}
    
    train_rows = []
    for _, r in m_df.tail(1200).iterrows(): 
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                score = 0
                adj = 1.0 if e_name == '실외' else (0.7 if e_name == '반실외' else 0.3)
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
# 2. 실시간 데이터 수집 (API)
# ==========================================================
tm, temp, humidity, rainfall, wind_speed = "-", "-", "-", "-", "-"
try:
    asos_params = {"serviceKey": SERVICE_KEY, "numOfRows": "1", "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR", "startDt": base_date, "startHh": base_hour, "endDt": base_date, "endHh": base_hour, "stnIds": "281"}
    res = requests.get("https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList", params=asos_params, timeout=10).json()
    item = res["response"]["body"]["items"]["item"][0]
    tm, temp, humidity, rainfall, wind_speed = item["tm"], item["ta"], item["hm"], item["rn"], item["ws"]
except: pass

pm10, pm25, o3, no2, co, so2, data_time = "-", "-", "-", "-", "-", "-", "-"
try:
    air_params = {"serviceKey": SERVICE_KEY, "returnType": "json", "numOfRows": "100", "sidoName": "경북", "ver": "1.0"}
    res = requests.get("https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty", params=air_params, timeout=10).json()
    target = next((i for i in res["response"]["body"]["items"] if "영천" in i["stationName"]), None)
    if target:
        pm10, pm25, o3, no2, co, so2, data_time = target["pm10Value"], target["pm25Value"], target["o3Value"], target["no2Value"], target["coValue"], target["so2Value"], target["dataTime"]
except: pass

# ==========================================================
# 3. UI 스타일 정의
# ==========================================================
card_style = "background-color:#f8f9fa; padding:22px; border-radius:20px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.05); height:350px;"
stat_card_style = "padding:20px; border-radius:15px; text-align:center; color:white; box-shadow:0 4px 10px rgba(0,0,0,0.1);"
title_style = "font-size:24px; font-weight:700; margin-bottom:14px; color:#1f2937;"
label_style = "font-size:14px; color:#6b7280; margin-bottom:4px;"
value_style = "font-size:22px; font-weight:700; color:#111827; margin-bottom:18px;"
time_style = "font-size:13px; color:#9ca3af; margin-top:12px; position:absolute; bottom:20px;"

# ==========================================================
# 4. 분석 엔진 (UI 표시 전 미리 계산하여 전역변수 확보)
# ==========================================================
heritage_df = pd.read_csv("data/processed/yc_heritage_feature.csv")
res_df = pd.DataFrame()

if ai_model:
    def safe_f(v):
        try: return float(v)
        except: return 0.0

    curr_env = {'temp': safe_f(temp), 'humidity': safe_f(humidity), 'rainfall': safe_f(rainfall), 'wind': safe_f(wind_speed), 'pm10': safe_f(pm10), 'pm25': safe_f(pm25), 'so2': safe_f(so2), 'no2': safe_f(no2), 'co': safe_f(co), 'o3': safe_f(o3)}
    mat_map = {'목조':0, '석조':1, '금속':2, '벽화':3}; exp_map = {'실외':0, '실내':1, '반실외':2}
    
    results = []
    for _, row in heritage_df.iterrows():
        m_code = mat_map.get(str(row['재질']).strip(), 4); e_code = exp_map.get(str(row['노출형태']).strip(), 0)
        input_v = pd.DataFrame([{**curr_env, 'mat_code': m_code, 'exp_code': e_code}])
        pred = ai_model.predict(input_v[feature_names])[0]; prob = ai_model.predict_proba(input_v[feature_names])[0]
        results.append({'문화재명': row['문화재명(국문)'], '재질': row['재질'], '노출': row['노출형태'], '위험수치': round(prob[2] * 100, 1), '등급': pred})
    
    res_df = pd.DataFrame(results)
    cnt_safe = len(res_df[res_df['등급']==0]); cnt_warn = len(res_df[res_df['등급']==1]); cnt_dang = len(res_df[res_df['등급']==2])
    
    # 전역 세션 스테이트에 위험 갯수 저장 (이후 상단 카드에서 참조)
    st.session_state['danger_count'] = cnt_dang

# ==========================================================
# 5. 메인 화면 구성 (UI)
# ==========================================================
st.markdown("<h1 style='font-size:30px;'>🏛 공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측</h1>", unsafe_allow_html=True)
st.divider()

st.markdown('<h3 style="font-size:22px; margin-bottom:15px;">🌿 실시간 영천 환경 지표 및 분석 요약</h3>', unsafe_allow_html=True)
left, center, right = st.columns([1.4, 2.0, 1.0])

with left:
    st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🌦 기상 환경</div><hr><div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px;">
    <div><div style="{label_style}">🌡 기온</div><div style="{value_style}">{temp} °C</div></div>
    <div><div style="{label_style}">💧 습도</div><div style="{value_style}">{humidity} %</div></div>
    <div><div style="{label_style}">🌧 강수량</div><div style="{value_style}">{rainfall} mm</div></div>
    <div><div style="{label_style}">💨 풍속</div><div style="{value_style}">{wind_speed} m/s</div></div>
    </div><div style="{time_style}">⏱ {tm}</div></div>""", unsafe_allow_html=True)

with center:
    st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🌫 대기오염 현황</div><hr><div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-top:20px;">
    <div><div style="{label_style}">PM10</div><div style="{value_style}">{pm10}</div><div style="{label_style}">O₃</div><div style="{value_style}">{o3}</div></div>
    <div><div style="{label_style}">PM2.5</div><div style="{value_style}">{pm25}</div><div style="{label_style}">NO₂</div><div style="{value_style}">{no2}</div></div>
    <div><div style="{label_style}">CO</div><div style="{value_style}">{co}</div><div style="{label_style}">SO₂</div><div style="{value_style}">{so2}</div></div>
    </div><div style="{time_style}">⏱ {data_time}</div></div>""", unsafe_allow_html=True)

with right:
    # 전역변수(st.session_state)에 저장된 위험 문화재 갯수를 카드에 표시
    st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🏛 문화재 현황</div><hr><div style="margin-top:20px;">
    <div style="{label_style}">분석 문화재 수</div><div style="{value_style}">{len(heritage_df)}개</div><br>
    <div style="{label_style}">🚨 고위험 문화재 (현재)</div><div style="{value_style}; color:#C62828;">{st.session_state['danger_count']}개</div></div>
    <div style="{time_style}">📍 경북 영천시</div></div>""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# 6. AI 분석 결과 카드 섹션
# ==========================================================
if not res_df.empty:
    st.markdown('<h3 style="font-size:22px; margin-bottom:15px;">📊 AI 위험도 판정 통계</h3>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div style="{stat_card_style} background-color:#2E7D32;"><h4>✅ 안전</h4><span style="font-size:30px; font-weight:bold;">{cnt_safe}</span> 건</div>', unsafe_allow_html=True)
    s2.markdown(f'<div style="{stat_card_style} background-color:#F9A825;"><h4>⚠️ 주의</h4><span style="font-size:30px; font-weight:bold;">{cnt_warn}</span> 건</div>', unsafe_allow_html=True)
    s3.markdown(f'<div style="{stat_card_style} background-color:#C62828;"><h4>🚨 위험</h4><span style="font-size:30px; font-weight:bold;">{st.session_state["danger_count"]}</span> 건</div>', unsafe_allow_html=True)

    st.write("")
    st.write("#### 🔎 상세 분석 리스트")
    st.dataframe(
        res_df.assign(판정=res_df['등급'].map({0:'안전', 1:'주의', 2:'위험'})).sort_values('위험수치', ascending=False),
        column_config={"위험수치": st.column_config.ProgressColumn("훼손 위험 지수", min_value=0, max_value=100, format="%f%%")},
        use_container_width=True, hide_index=True
    )
else:
    st.error("분석 데이터를 불러오지 못했습니다.")

st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
