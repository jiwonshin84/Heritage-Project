import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# 0. 설정 및 초기화
# ==========================================================
st.set_page_config(
    page_title="영천 문화재 AI 학습 데이터 분석",
    page_icon="📊",
    layout="wide"
)

# 한글 변수명 매핑
KOR_NAMES = {
    'temp': '기온', 'humidity': '습도', 'rainfall': '강수량', 'wind': '풍속',
    'pm10': '미세먼지(PM10)', 'pm25': '초미세먼지(PM2.5)', 'so2': '아황산가스',
    'no2': '이산화질소', 'co': '일산화탄소', 'o3': '오존',
    'mat_code': '재질분류', 'exp_code': '노출분류', 'target': '위험등급'
}

# ==========================================================
# 1. 모델 학습 및 데이터 처리 로직
# ==========================================================
@st.cache_resource
def get_training_report_data():
    weather_path = "data/processed/[2016_2025] yeongcheon_weather_daily.csv"
    air_path = "data/processed/[2019_2025] air_quality.csv"
    
    if not os.path.exists(weather_path) or not os.path.exists(air_path):
        return None, None, None

    w_df = pd.read_csv(weather_path)
    a_df = pd.read_csv(air_path)
    w_df = w_df.rename(columns={
        'avg_temperature_c': 'temp', 'daily_precipitation_mm': 'rainfall',
        'avg_wind_speed_ms': 'wind', 'avg_relative_humidity_pct': 'humidity'
    })
    m_df = pd.merge(w_df, a_df, on='date', how='inner')

    mats = {'목조': 0, '석조': 1, '금속': 2, '벽화': 3, '기타': 4}
    exps = {'실외': 0, '실내': 1, '반실외': 2}
    
    train_rows = []
    # 1800행 규모의 데이터 생성을 위해 tail 조정
    for _, r in m_df.tail(150).iterrows(): 
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
    
    # 모델 학습 (중요도 계산용)
    features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'so2', 'no2', 'co', 'o3', 'mat_code', 'exp_code']
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(tdf[features].fillna(0), tdf['target'])
    
    return model, features, tdf

ai_model, feature_names, tdf_raw = get_training_report_data()

# ==========================================================
# 2. 화면 구성: 📊 AI 모델 학습 리포트 (단일 화면)
# ==========================================================
st.markdown("<h1 style='font-size:30px;'>📊 AI 모델 학습 및 데이터 분석 리포트</h1>", unsafe_allow_html=True)
st.write("본 리포트는 영천시 문화재 훼손 위험 예측을 위해 사용된 AI 모델의 학습 결과와 데이터 중복 현황을 분석합니다.")
st.divider()

if ai_model is not None and tdf_raw is not None:
    # --- 상단: 알고리즘 분석 ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 환경 요인별 중요도")
        importances = ai_model.feature_importances_
        kor_features = [KOR_NAMES.get(f, f) for f in feature_names]
        feat_df = pd.DataFrame({'요인': kor_features, '중요도': importances}).set_index('요인').sort_values('중요도', ascending=True)
        st.bar_chart(feat_df, color="#4A90E2")
        st.info("💡 AI가 위험 등급을 판단할 때 가장 결정적인 영향을 준 환경 요소 순위입니다.")

    with col2:
        st.subheader("📝 학습 모델 정보")
        st.write(f"- **총 학습 데이터**: {len(tdf_raw):,}행")
        st.write(f"- **사용한 알고리즘**: Random Forest Classifier")
        st.write(f"- **분석 목적**: 기상 및 대기오염에 따른 재질별 훼손 확률 예측")
        st.write("---")
        st.markdown("""
        **재질/노출분류 기준:**
        - 목조(0), 석조(1), 금속(2), 벽화(3), 기타(4)
        - 실외(0), 실내(1), 반실외(2)
        """)

    st.divider()

    # --- 하단: 환경/미세먼지 동일 데이터 추출 (요청 사항) ---
    st.subheader("🔍 환경 및 대기 데이터 중복 분석")
    st.write("학습 데이터 중 기상 정보(기온, 습도, 강수, 풍속)와 미세먼지 정보(PM10, PM2.5)가 완벽히 일치하는 행들만 추출한 결과입니다.")
    
    # 중복 체크할 컬럼 선정
    dup_columns = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25']
    
    # 중복된 데이터만 필터링 (keep=False를 통해 중복되는 모든 행 표시)
    duplicates = tdf_raw[tdf_raw.duplicated(subset=dup_columns, keep=False)]
    
    # 한글 변명으로 변경하여 표시
    duplicates_kor = duplicates.rename(columns=KOR_NAMES)
    
    if not duplicates_kor.empty:
        st.write(f"✅ 총 **{len(duplicates_kor):,}**개의 중복 행이 발견되었습니다.")
        # 전체 행을 볼 수 있도록 dataframe 사용 (스크롤 가능)
        st.dataframe(duplicates_kor, use_container_width=True, height=600)
    else:
        st.info("데이터 중 기상 및 미세먼지 정보가 완벽히 일치하는 중복 행이 존재하지 않습니다.")

else:
    st.error("데이터 로드 실패. 'data' 폴더 내에 필요한 기상/대기 CSV 파일이 있는지 확인해주세요.")

# 하단 캡션
st.divider()
st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
