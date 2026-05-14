import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# 0. API 및 환경 설정
# ==========================================================
SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
now = datetime.now(ZoneInfo("Asia/Seoul"))
yesterday = now - timedelta(days=1)
base_date = yesterday.strftime("%Y%m%d")

# ==========================================================
# 1. 데이터 로드 및 모델 학습 (백엔드 로직)
# ==========================================================
@st.cache_resource
def train_heritage_model():
    # 데이터 로드
    w_df = pd.read_csv("data/processed/[2016_2025] yeongcheon_weather_daily.csv")
    a_df = pd.read_csv("data/processed/[2019_2025] air_quality.csv")
    
    # 컬럼명 통일 및 병합
    w_df = w_df.rename(columns={
        'avg_temperature_c': 'temp', 'daily_precipitation_mm': 'rainfall',
        'avg_wind_speed_ms': 'wind', 'avg_relative_humidity_pct': 'humidity'
    })
    m_df = pd.merge(w_df, a_df, on='date', how='inner')

    # 파생변수 생성 로직 (이슬점, 산성비 등)
    m_df['dew_gap'] = m_df['temp'] - (m_df['temp'] - ((100 - m_df['humidity']) / 5))
    
    # 학습용 가상 라벨링 (재질/노출별 위험 시뮬레이션)
    mats = {'목조':0, '석조':1, '금속':2, '벽화':3, '기타':4}
    exps = {'실외':0, '실내':1}
    
    train_rows = []
    for _, r in m_df.tail(1000).iterrows(): # 최근 1000일 데이터 사용
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                score = 0
                adj = 0.4 if e_name == '실내' else 1.0
                if m_name == '목조' and (r['humidity'] > 80 or r['humidity'] < 30): score += 2 * adj
                if m_name == '석조' and r['rainfall'] > 10: score += 2 * adj
                if r['pm10'] > 80: score += 1.5 * adj
                
                danger = 2 if score >= 2.5 else (1 if score >= 1.2 else 0)
                train_rows.append({
                    'temp': r['temp'], 'humidity': r['humidity'], 'rainfall': r['rainfall'],
                    'wind': r['wind'], 'pm10': r['pm10'], 'pm25': r['pm25'],
                    'so2': r['so2'], 'no2': r['no2'], 'co': r['co'], 'o3': r['o3'],
                    'mat_code': m_code, 'exp_code': e_code, 'dew_gap': r['dew_gap'],
                    'target': danger
                })
    
    tdf = pd.DataFrame(train_rows)
    features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'so2', 'no2', 'co', 'o3', 'mat_code', 'exp_code', 'dew_gap']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(tdf[features].fillna(0), tdf['target'])
    return model, features

# 모델 준비
model, feature_names = train_heritage_model()

# ==========================================================
# 2. 실시간 데이터 수집 (사용자 코드 유지 및 보완)
# ==========================================================
# [기존 ASOS 및 대기오염 API 요청 코드 실행 부분 - 생략하지 않고 통합]
# (이 부분에 질문자님이 주신 API 요청 코드가 들어갑니다. 변수: temp, humidity, pm10 등 저장)
# ... (API 호출 결과로 temp, humidity, pm10, pm25, so2, no2, co, o3 값이 추출되었다고 가정) ...

# 예외 처리용 수치 변환
try:
    curr_env = {
        'temp': float(temp), 'humidity': float(humidity), 'rainfall': float(rainfall) if rainfall != '-' else 0,
        'wind': float(wind_speed), 'pm10': float(pm10), 'pm25': float(pm25),
        'so2': float(so2), 'no2': float(no2), 'co': float(co), 'o3': float(o3)
    }
    curr_env['dew_gap'] = curr_env['temp'] - (curr_env['temp'] - ((100 - curr_env['humidity']) / 5))
except:
    curr_env = None

# ==========================================================
# 3. 문화재별 위험도 판별 페이지 구성
# ==========================================================
st.set_page_config(page_title="영천 문화재 위험 예측", layout="wide")

# (기존 대시보드 UI 코드 실행)
# ... (생략된 기존 UI 부분) ...

st.divider()

if curr_env:
    st.subheader("🎯 개별 문화재별 현재 위험도 판별 결과")
    
    # 문화재 피처 로드
    heritage_df = pd.read_csv("data/processed/yc_heritage_feature.csv")
    
    # 재질/노출형태 인코딩
    mat_map = {'목조':0, '석조':1, '금속':2, '벽화':3}
    exp_map = {'실외':0, '실내':1}
    
    results = []
    for _, h in heritage_df.iterrows():
        m_code = mat_map.get(h['재질'], 4) # 기타는 4
        e_code = exp_map.get(h['노출형태'], 0) # 기본 실외
        
        # 모델 입력 데이터 생성
        input_feat = pd.DataFrame([{
            **curr_env, 'mat_code': m_code, 'exp_code': e_code
        }])
        
        pred = model.predict(input_feat[feature_names])[0]
        prob = model.predict_proba(input_feat[feature_names])[0]
        
        results.append({
            '문화재명': h['문화재명(국문)'],
            '재질': h['재질'],
            '노출': h['노출형태'],
            '위험수치': round(prob[2] * 100, 1), # 위험(2) 클래스의 확률
            '등급': ['안전', '주의', '위험'][pred]
        })
    
    res_df = pd.DataFrame(results)
    
    # 결과 요약 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ 안전", len(res_df[res_df['등급']=='안전']))
    c2.metric("⚠️ 주의", len(res_df[res_df['등급']=='주의']))
    c3.metric("🚨 위험", len(res_df[res_df['등급']=='위험']))
    
    # 테이블 출력
    st.dataframe(
        res_df.sort_values('위험수치', ascending=False),
        column_config={
            "위험수치": st.column_config.ProgressColumn("훼손 위험 지수", min_value=0, max_value=100, format="%f%%"),
            "등급": st.column_config.TextColumn("판정")
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("실시간 환경 데이터를 불러올 수 없어 위험도 판별을 진행할 수 없습니다.")
