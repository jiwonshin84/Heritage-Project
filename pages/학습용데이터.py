import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# 0. 페이지 설정
# ==========================================================
st.set_page_config(
    page_title="영천 문화재 AI 학습 리포트",
    page_icon="📊",
    layout="wide"
)

# 한글 변수명 매핑
KOR_NAMES = {
    'temp': '기온', 'humidity': '습도', 'rainfall': '강수량', 'wind': '풍속',
    'pm10': '미세먼지(PM10)', 'pm25': '초미세먼지(PM2.5)', 'so2': '아황산가스',
    'no2': '이산화질소', 'co': '일산화탄소', 'o3': '오존',
    'mat_code': '재질분류', 'exp_code': '노출분류', 
    'danger_score': '위험점수(파생)', 'target': '위험등급'
}

# ==========================================================
# 1. 데이터 생성 및 파생변수 로직 (AI 학습 리포트용)
# ==========================================================
@st.cache_resource
def get_advanced_training_data():
    # 실제 경로에 파일이 없을 경우를 대비한 샘플 데이터 구성 로직 포함
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

    # 분류 체계 정의
    mats = {'목조': 0, '석조': 1, '금속': 2, '벽화': 3, '기타': 4}
    exps = {'실외': 0, '실내': 1, '반실외': 2}
    
    train_rows = []
    # 약 1,800행 생성을 위해 원본 데이터의 최근 120일치 활용 (120일 * 5재질 * 3노출 = 1,800행)
    for _, r in m_df.tail(120).iterrows(): 
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                
                # --- [파생 변수: 위험 점수(danger_score) 계산 로직] ---
                score = 0.0
                
                # 1. 노출도 가중치 (실외일수록 환경 영향 극대화)
                exposure_weight = 1.0 if e_name == '실외' else (0.6 if e_name == '반실외' else 0.2)
                
                # 2. 재질별 기상 영향
                if m_name == '목조':
                    if r['humidity'] > 80: score += 3.0  # 부패 위험
                    if r['humidity'] < 30: score += 2.0  # 건조 균열 위험
                elif m_name == '석조':
                    if r['rainfall'] > 20: score += 2.5  # 수분 침투 훼손
                elif m_name == '금속':
                    if r['humidity'] > 70 and r['temp'] > 25: score += 2.0  # 부식 가속
                
                # 3. 대기오염 공통 영향 (미세먼지 및 산성비 요인)
                if r['pm10'] > 80: score += 1.5
                if r['o3'] > 0.06: score += 1.0
                
                # 가중치 적용 최종 점수 생성
                final_score = round(score * exposure_weight, 2)
                
                # 최종 등급 결정 (0:안전, 1:주의, 2:위험)
                danger_level = 2 if final_score >= 2.5 else (1 if final_score >= 1.0 else 0)
                
                train_rows.append({
                    'temp': r['temp'], 'humidity': r['humidity'], 'rainfall': r['rainfall'],
                    'wind': r['wind'], 'pm10': r['pm10'], 'pm25': r['pm25'],
                    'so2': r['so2'], 'no2': r['no2'], 'co': r['co'], 'o3': r['o3'],
                    'mat_code': m_code, 'exp_code': e_code, 
                    'danger_score': final_score, 'target': danger_level
                })
    
    tdf = pd.DataFrame(train_rows)
    
    # AI 모델 학습
    features = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25', 'so2', 'no2', 'co', 'o3', 'mat_code', 'exp_code']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(tdf[features].fillna(0), tdf['target'])
    
    return model, features, tdf

ai_model, feature_names, tdf_raw = get_advanced_training_data()

# ==========================================================
# 2. 화면 구성: AI 모델 학습 리포트
# ==========================================================
st.title("📊 AI 학습 리포트: 파생변수 기반 위험도 분석")
st.markdown("기상/대기 정보에 **재질분류**와 **노출형태**를 결합한 **위험 점수(Danger Score)** 파생 변수를 활용하여 학습을 수행했습니다.")
st.divider()

if ai_model is not None and tdf_raw is not None:
    
    # --- 섹션 1: 환경 요인 중요도 분석 ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 요인별 학습 중요도")
        importances = ai_model.feature_importances_
        feat_df = pd.DataFrame({
            '요인': [KOR_NAMES.get(f, f) for f in feature_names],
            '중요도': importances
        }).set_index('요인').sort_values('중요도', ascending=True)
        
        st.bar_chart(feat_df, color="#4A90E2")
        st.caption("AI가 등급을 분류할 때 가장 비중 있게 다룬 변수들입니다.")

    with col2:
        st.subheader("⚙️ 파생 변수 생성 로직")
        st.info("""
        **위험 점수(Danger Score) 산출식:**
        1. **기본점수**: 재질별 특정 임계치(습도 80%↑, 강수 20mm↑ 등) 초과 시 부여
        2. **오염점수**: 미세먼지(PM10) 및 오존 농도에 따른 추가 점수
        3. **보정치(가중치)**: 
           - 실외(1.0) / 반실외(0.6) / 실내(0.2)
        4. **최종등급**: 점수 2.5 이상 시 '위험(2)'으로 레이블링
        """)

    st.divider()

    # --- 섹션 2: 중복 데이터 분석 (전체 데이터 공개) ---
    st.subheader("🔍 환경 조건이 동일한 학습 데이터 상세 (전체)")
    st.write("학습 데이터 중 기상 및 미세먼지 수치가 동일한 행들을 추출하여 재질/노출에 따른 등급 변화를 확인합니다.")

    # 중복 기준 컬럼 (환경 변수들)
    dup_criteria = ['temp', 'humidity', 'rainfall', 'wind', 'pm10', 'pm25']
    
    # 환경 데이터가 똑같은 데이터 필터링 (전체 데이터 중 중복 건 포함)
    all_duplicates = tdf_raw[tdf_raw.duplicated(subset=dup_criteria, keep=False)]
    
    # 한글 변수명 적용 및 정렬
    display_df = all_duplicates.rename(columns=KOR_NAMES).sort_values(
        by=['기온', '습도', '미세먼지(PM10)']
    )

    if not display_df.empty:
        st.write(f"✅ 조건이 일치하는 중복 환경 데이터: 총 **{len(display_df):,}**행")
        # 전체 행 표시 (height를 크게 설정하여 스크롤로 전체 확인 가능)
        st.dataframe(display_df, use_container_width=True, height=800)
    else:
        st.info("환경 정보가 완벽히 일치하는 데이터가 없습니다. 전체 학습 데이터를 대신 표시합니다.")
        st.dataframe(tdf_raw.rename(columns=KOR_NAMES), use_container_width=True)

else:
    st.error("데이터 파일을 찾을 수 없습니다. 경로와 파일명을 확인해주세요.")

# 하단 정보
st.divider()
st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
