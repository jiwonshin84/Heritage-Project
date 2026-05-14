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
    page_title="영천 지역 문화재 훼손 위험 예측",
    page_icon="🏛",
    layout="wide"
)

# 전역 변수(위험 문화재 수) 초기화
if 'danger_count' not in st.session_state:
    st.session_state['danger_count'] = 0

SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"

# 한글 변수명 매핑 딕셔너리
KOR_NAMES = {
    'temp': '기온', 'humidity': '습도', 'rainfall': '강수량', 'wind': '풍속',
    'pm10': '미세먼지(PM10)', 'pm25': '초미세먼지(PM2.5)', 'so2': '아황산가스',
    'no2': '이산화질소', 'co': '일산화탄소', 'o3': '오존',
    'mat_code': '재질분류', 'exp_code': '노출분류', 'target': '위험등급'
}

# ==========================================================
# 1. 모델 학습 로직
# ==========================================================
@st.cache_resource
def train_heritage_model():
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
    
    # 학습 데이터 컬럼 한글화
    tdf_kor = tdf.rename(columns=KOR_NAMES)
    return model, features, tdf_kor

ai_model, feature_names, training_data = train_heritage_model()

# ==========================================================
# 2. 실시간 환경 데이터 수집
# ==========================================================
now = datetime.now(ZoneInfo("Asia/Seoul"))
yesterday = now - timedelta(days=1)
base_date = yesterday.strftime("%Y%m%d")
base_hour = "23"

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
# 3. 사이드바 및 AI 분석
# ==========================================================
with st.sidebar:
    st.title("📌 메뉴")
    menu = st.radio("이동할 화면을 선택하세요", ["🏠 메인 대시보드", "📊 AI 모델 학습 리포트"])
    st.divider()
    st.info("선화여고 - 영천 헤리티지 AI 탐구단")

# AI 분석 수행 및 위험 개수 산출
df_main = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
if ai_model:
    def safe_f(v):
        try: return float(v)
        except: return 0.0
    curr_env = {'temp': safe_f(temp), 'humidity': safe_f(humidity), 'rainfall': safe_f(rainfall), 'wind': safe_f(wind_speed), 'pm10': safe_f(pm10), 'pm25': safe_f(pm25), 'so2': safe_f(so2), 'no2': safe_f(no2), 'co': safe_f(co), 'o3': safe_f(o3)}
    mat_map = {'목조':0, '석조':1, '금속':2, '벽화':3}; exp_map = {'실외':0, '실내':1, '반실외':2}
    
    results = []
    for _, row in df_main.iterrows():
        m_code = mat_map.get(str(row.get('재질','')).strip(), 4); e_code = exp_map.get(str(row.get('노출형태','')).strip(), 0)
        input_v = pd.DataFrame([{**curr_env, 'mat_code': m_code, 'exp_code': e_code}])
        pred = ai_model.predict(input_v[feature_names])[0]; prob = ai_model.predict_proba(input_v[feature_names])[0]
        results.append({'문화재명': row['문화재명(국문)'], '재질': row.get('재질','-'), '노출': row.get('노출형태','-'), '위험수치': round(prob[2] * 100, 1), '등급': pred})
    
    res_df = pd.DataFrame(results)
    st.session_state['danger_count'] = len(res_df[res_df['등급']==2])
    cnt_safe = len(res_df[res_df['등급']==0]); cnt_warn = len(res_df[res_df['등급']==1])

# ==========================================================
# 4. 화면 구성: 🏠 메인 대시보드
# ==========================================================
if menu == "🏠 메인 대시보드":
    st.markdown("<h1 style='font-size:30px;'>🏛 영천 지역 문화재 훼손 위험 예측</h1>", unsafe_allow_html=True)
    st.divider()

    # 상단 환경 카드
    left, center, right = st.columns([1.4, 2.0, 1.0])
    card_style = "background-color:#f8f9fa; padding:22px; border-radius:20px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.05); height:350px;"
    
    with left:
        st.markdown(f"""<div style="{card_style}; position:relative;"><h3>🌦 기상 환경</h3><hr>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px;">
        <b>🌡 기온</b>: {temp} °C<br><b>💧 습도</b>: {humidity} %<br><b>🌧 강수량</b>: {rainfall} mm<br><b>💨 풍속</b>: {wind_speed} m/s
        </div><div style="font-size:12px; color:gray; position:absolute; bottom:20px;">⏱ {tm}</div></div>""", unsafe_allow_html=True)

    with center:
        st.markdown(f"""<div style="{card_style}; position:relative;"><h3>🌫 대기오염</h3><hr>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-top:20px;">
        <div>PM10: {pm10}<br>O₃: {o3}</div><div>PM2.5: {pm25}<br>NO₂: {no2}</div><div>CO: {co}<br>SO₂: {so2}</div>
        </div><div style="font-size:12px; color:gray; position:absolute; bottom:20px;">⏱ {data_time}</div></div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""<div style="{card_style}; position:relative;"><h3>🏛 현황 요약</h3><hr>
        <div style="margin-top:20px;">분석 대상: {len(df_main)}개<br><br><b>🚨 고위험 대상</b><br><span style="font-size:24px; color:red;">{st.session_state['danger_count']}개</span></div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    # 통계 및 리스트
    s1, s2, s3 = st.columns(3)
    s1.metric("안전", f"{cnt_safe} 건")
    s2.metric("주의", f"{cnt_warn} 건")
    s3.metric("위험", f"{st.session_state['danger_count']} 건")
    
    st.dataframe(res_df.assign(판정=res_df['등급'].map({0:'안전', 1:'주의', 2:'위험'})).sort_values('위험수치', ascending=False), use_container_width=True, hide_index=True)

# ==========================================================
# 5. 화면 구성: 📊 AI 모델 학습 리포트
# ==========================================================
else:
    st.markdown("<h1 style='font-size:30px;'>📊 AI 모델 학습 및 알고리즘 리포트</h1>", unsafe_allow_html=True)
    st.divider()

    if ai_model is not None:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("🎯 환경 요인별 영향도 (중요도)")
            # 중요도 데이터 한글화 및 차트용 정렬
            importances = ai_model.feature_importances_
            kor_features = [KOR_NAMES.get(f, f) for f in feature_names]
            feat_df = pd.DataFrame({'요인': kor_features, '중요도': importances}).set_index('요인').sort_values('중요도', ascending=True)
            
            # Streamlit 내장 차트 사용 (한글 깨짐 없음)
            st.bar_chart(feat_df, color="#4A90E2")
            st.info("💡 차트의 값이 높을수록 AI가 위험 등급을 결정할 때 더 많이 참고한 환경 요소입니다.")

        with col2:
            st.subheader("📚 학습 데이터 샘플 (한글 변수명)")
            st.write(f"영천시의 과거 데이터를 기반으로 생성된 {len(training_data):,}행의 학습 데이터 중 일부입니다.")
            st.dataframe(training_data.head(15), use_container_width=True)
            
            with st.expander("📝 학습 로직 및 코드값 안내"):
                st.write("""
                - **재질분류**: 목조(0), 석조(1), 금속(2), 벽화(3), 기타(4)
                - **노출분류**: 실외(0), 실내(1), 반실외(2)
                - **위험등급**: 안전(0), 주의(1), 위험(2)
                """)
    else:
        st.error("데이터를 찾을 수 없습니다. 'data' 폴더 내 CSV 파일들을 확인해주세요.")

st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
