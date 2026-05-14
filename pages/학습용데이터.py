import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# 0. 페이지 설정 및 초기화
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

# ==========================================================
# 1. AI 모델 학습 로직
# ==========================================================
@st.cache_resource
def train_heritage_model():
    # 파일 경로 설정 (사용자 환경에 맞게 조정 필요)
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
    # 데이터가 너무 많을 경우를 대비해 최근 1200일 데이터 학습
    for _, r in m_df.tail(1200).iterrows(): 
        for m_name, m_code in mats.items():
            for e_name, e_code in exps.items():
                score = 0
                adj = 1.0 if e_name == '실외' else (0.7 if e_name == '반실외' else 0.3)
                
                # 가중치 알고리즘
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
    return model, features, tdf

ai_model, feature_names, training_data = train_heritage_model()

# ==========================================================
# 2. 실시간 환경 데이터 수집 (API)
# ==========================================================
now = datetime.now(ZoneInfo("Asia/Seoul"))
yesterday = now - timedelta(days=1)
base_date = yesterday.strftime("%Y%m%d")
base_hour = "23"

# [기상 데이터 수집]
tm, temp, humidity, rainfall, wind_speed = "-", "-", "-", "-", "-"
try:
    asos_params = {"serviceKey": SERVICE_KEY, "numOfRows": "1", "dataType": "JSON", "dataCd": "ASOS", "dateCd": "HR", "startDt": base_date, "startHh": base_hour, "endDt": base_date, "endHh": base_hour, "stnIds": "281"}
    res = requests.get("https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList", params=asos_params, timeout=10).json()
    item = res["response"]["body"]["items"]["item"][0]
    tm, temp, humidity, rainfall, wind_speed = item["tm"], item["ta"], item["hm"], item["rn"], item["ws"]
except: pass

# [대기 데이터 수집]
pm10, pm25, o3, no2, co, so2, data_time = "-", "-", "-", "-", "-", "-", "-"
try:
    air_params = {"serviceKey": SERVICE_KEY, "returnType": "json", "numOfRows": "100", "sidoName": "경북", "ver": "1.0"}
    res = requests.get("https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty", params=air_params, timeout=10).json()
    target = next((i for i in res["response"]["body"]["items"] if "영천" in i["stationName"]), None)
    if target:
        pm10, pm25, o3, no2, co, so2, data_time = target["pm10Value"], target["pm25Value"], target["o3Value"], target["no2Value"], target["coValue"], target["so2Value"], target["dataTime"]
except: pass

# ==========================================================
# 3. 사이드바 메뉴
# ==========================================================
with st.sidebar:
    st.title("📌 메뉴")
    menu = st.radio("이동할 화면을 선택하세요", ["🏠 메인 대시보드", "📊 AI 모델 학습 리포트"])
    st.divider()
    st.caption("제6회 학생 SW·AI 인재양성 프로젝트")
    st.caption("선화여고 - 영천 헤리티지 AI 탐구단")

# ==========================================================
# 4. 화면 로직 - AI 사전 분석 (전역변수 확보용)
# ==========================================================
df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")

if ai_model is not None:
    def safe_f(v):
        try: return float(v)
        except: return 0.0

    curr_env = {
        'temp': safe_f(temp), 'humidity': safe_f(humidity), 'rainfall': safe_f(rainfall),
        'wind': safe_f(wind_speed), 'pm10': safe_f(pm10), 'pm25': safe_f(pm25),
        'so2': safe_f(so2), 'no2': safe_f(no2), 'co': safe_f(co), 'o3': safe_f(o3)
    }
    
    mat_map = {'목조': 0, '석조': 1, '금속': 2, '벽화': 3}
    exp_map = {'실외': 0, '실내': 1, '반실외': 2}
    
    analysis_results = []
    for _, row in df.iterrows():
        m_code = mat_map.get(str(row.get('재질', '')).strip(), 4)
        e_code = exp_map.get(str(row.get('노출형태', '')).strip(), 0)
        input_data = pd.DataFrame([{**curr_env, 'mat_code': m_code, 'exp_code': e_code}])
        pred = ai_model.predict(input_data[feature_names])[0]
        prob = ai_model.predict_proba(input_data[feature_names])[0]
        analysis_results.append({
            '문화재명': row['문화재명(국문)'], '재질': row.get('재질','-'), '노출': row.get('노출형태','-'),
            '위험수치': round(prob[2] * 100, 1), '등급': pred
        })
    
    res_df = pd.DataFrame(analysis_results)
    st.session_state['danger_count'] = len(res_df[res_df['등급'] == 2])
    cnt_safe = len(res_df[res_df['등급'] == 0])
    cnt_warn = len(res_df[res_df['등급'] == 1])

# ==========================================================
# 5-1. 화면 구성: 🏠 메인 대시보드
# ==========================================================
if menu == "🏠 메인 대시보드":
    st.markdown("<h1 style='font-size:30px;'>🏛 공공 환경 데이터 기반 영천 지역 문화재 훼손 위험 예측</h1>", unsafe_allow_html=True)
    st.markdown("영천 지역 문화재와 공공 환경데이터를 분석하여 문화재 훼손 위험을 사전에 예측합니다.")
    st.divider()

    # 상단 환경 카드 섹션
    st.markdown('<h3 style="font-size:25px; margin-bottom:10px;">🌿 영천시 환경 데이터 및 문화재 현황</h3>', unsafe_allow_html=True)
    left, center, right = st.columns([1.4, 2.0, 1.0])

    card_style = "background-color:#f8f9fa; padding:22px; border-radius:20px; border:1px solid #e5e7eb; box-shadow:0 4px 12px rgba(0,0,0,0.05); height:350px;"
    title_style = "font-size:24px; font-weight:700; margin-bottom:14px; color:#1f2937;"
    label_style = "font-size:14px; color:#6b7280; margin-bottom:4px;"
    value_style = "font-size:22px; font-weight:700; color:#111827; margin-bottom:18px;"
    time_style = "font-size:13px; color:#9ca3af; margin-top:12px; position:absolute; bottom:20px;"

    with left:
        st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🌦 기상 환경</div><hr><div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px;">
        <div><div style="{label_style}">🌡 기온</div><div style="{value_style}">{temp} °C</div></div>
        <div><div style="{label_style}">💧 습도</div><div style="{value_style}">{humidity} %</div></div>
        <div><div style="{label_style}">🌧 강수량</div><div style="{value_style}">{rainfall} mm</div></div>
        <div><div style="{label_style}">💨 풍속</div><div style="{value_style}">{wind_speed} m/s</div></div>
        </div><div style="{time_style}">⏱ 측정 시각 : {tm}</div></div>""", unsafe_allow_html=True)

    with center:
        st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🌫 대기오염 현황</div><hr><div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; margin-top:20px;">
        <div><div style="{label_style}">PM10</div><div style="{value_style}">{pm10}</div><div style="{label_style}">O₃</div><div style="{value_style}">{o3}</div></div>
        <div><div style="{label_style}">PM2.5</div><div style="{value_style}">{pm25}</div><div style="{label_style}">NO₂</div><div style="{value_style}">{no2}</div></div>
        <div><div style="{label_style}">CO</div><div style="{value_style}">{co}</div><div style="{label_style}">SO₂</div><div style="{value_style}">{so2}</div></div>
        </div><div style="{time_style}">⏱ 측정 시각 : {data_time}</div></div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""<div style="{card_style}; position:relative;"><div style="{title_style}">🏛 문화재 현황</div><hr><div style="margin-top:20px;">
        <div style="{label_style}">분석 문화재 수</div><div style="{value_style}">{len(df)}개</div><br>
        <div style="{label_style}">⚠ 고위험 문화재</div><div style="{value_style}; color:#C62828;">{st.session_state['danger_count']}개</div>
        </div><div style="{time_style}">📍 경북 영천시</div></div>""", unsafe_allow_html=True)

    st.divider()

    # 중단 통계 카드
    st.markdown('<h3 style="font-size:22px; margin-bottom:15px;">📊 AI 위험도 판정 통계</h3>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    stat_style = "padding:20px; border-radius:15px; text-align:center; color:white; box-shadow:0 4px 10px rgba(0,0,0,0.1);"
    s1.markdown(f'<div style="{stat_style} background-color:#2E7D32;"><h4>✅ 안전</h4><span style="font-size:30px; font-weight:bold;">{cnt_safe}</span> 건</div>', unsafe_allow_html=True)
    s2.markdown(f'<div style="{stat_style} background-color:#F9A825;"><h4>⚠️ 주의</h4><span style="font-size:30px; font-weight:bold;">{cnt_warn}</span> 건</div>', unsafe_allow_html=True)
    s3.markdown(f'<div style="{stat_style} background-color:#C62828;"><h4>🚨 위험</h4><span style="font-size:30px; font-weight:bold;">{st.session_state["danger_count"]}</span> 건</div>', unsafe_allow_html=True)

    # 하단 분석 리스트
    st.write("")
    st.dataframe(
        res_df.assign(판정=res_df['등급'].map({0:'안전', 1:'주의', 2:'위험'})).sort_values('위험수치', ascending=False),
        column_config={"위험수치": st.column_config.ProgressColumn("훼손 위험 지수", min_value=0, max_value=100, format="%f%%")},
        use_container_width=True, hide_index=True
    )

# ==========================================================
# 5-2. 화면 구성: 📊 AI 모델 학습 리포트
# ==========================================================
else:
    st.markdown("<h1 style='font-size:30px;'>📊 AI 모델 학습 및 데이터 요약</h1>", unsafe_allow_html=True)
    st.write("AI가 영천시의 과거 환경 데이터를 기반으로 위험도를 예측하는 원리를 보여줍니다.")
    st.divider()

    if ai_model is not None:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("🎯 환경 요인별 중요도")
            importances = ai_model.feature_importances_
            feat_df = pd.DataFrame({'요소': feature_names, '중요도': importances}).sort_values('중요도', ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(feat_df['요소'], feat_df['중요도'], color='#4A90E2')
            st.pyplot(fig)
            st.info("※ 중요도가 높을수록 위험도 판정에 더 큰 영향을 미칩니다.")

        with c2:
            st.subheader("📂 학습 데이터 샘플")
            st.write(f"총 {len(training_data):,}개의 환경-재질 시뮬레이션 데이터를 학습했습니다.")
            st.dataframe(training_data.head(15), use_container_width=True)
            
            st.markdown("""
            **💡 AI 학습 알고리즘 정보**
            - **모델**: Random Forest (앙상블 결정 트리)
            - **학습 데이터**: 영천시 과거 기상 및 대기오염 관측값
            - **보정치**: 실외(100%), 반실외(70%), 실내(30%) 노출 등급 반영
            """)
    else:
        st.warning("학습 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")

# 공통 하단 캡션
st.divider()
st.caption("제6회 학생 SW·AI 인재양성 프로젝트 | 선화여고 - 영천 헤리티지 AI 탐구단")
