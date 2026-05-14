import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# ============================================================
# 0. 페이지 설정
# ============================================================
st.set_page_config(page_title="영천 문화재 훼손 위험 예측", layout="wide")

st.title("🏛️ 영천 지역 문화재 훼손 위험 실시간 예측")

# ============================================================
# 1. 데이터 수집 함수 (KeyError 방지 로직 추가)
# ============================================================
@st.cache_data(ttl=3600)
def get_latest_environment_data():
    ASOS_SERVICE_KEY = "feb2bfabd299d5d05e89c7aec49ba7e706112603e76549a92e868bd86ec60323"
    
    # 오늘 데이터가 아직 안 올라왔을 수 있으므로 최근 3일치 조회
    end_dt = datetime.now().strftime('%Y%m%d')
    start_dt = (datetime.now() - timedelta(days=3)).strftime('%Y%m%d')
    
    url = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "10",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_dt,
        "endDt": end_dt,
        "stnIds": "281"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        res_json = response.json()
        
        # 데이터 존재 여부 확인 (중요!)
        if "response" in res_json and "body" in res_json["response"] and "items" in res_json["response"]["body"]:
            items = res_json["response"]["body"]["items"]["item"]
            if not items:
                st.error("기상청 API에 최근 데이터가 아직 등록되지 않았습니다.")
                return pd.DataFrame()
            
            weather_df = pd.DataFrame(items)
            
            # 필요한 컬럼이 있는지 확인 후 필터링
            required_cols = ["tm", "avgTa", "avgRhm", "sumRn", "avgWs"]
            if all(col in weather_df.columns for col in required_cols):
                weather_df = weather_df[required_cols].copy()
                weather_df.columns = ["date", "temp", "humidity", "rainfall", "wind"]
                
                # 타입 변환
                for col in ["temp", "humidity", "rainfall", "wind"]:
                    weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")
                weather_df["rainfall"] = weather_df["rainfall"].fillna(0)
                weather_df["date"] = pd.to_datetime(weather_df["date"])
                
                # 미세먼지 데이터 병합
                air_path = "data/processed/[2019_2025] air_quality.csv"
                if os.path.exists(air_path):
                    air_df = pd.read_csv(air_path)
                    air_df["date"] = pd.to_datetime(air_df["date"])
                    combined = pd.merge(weather_df, air_df, on="date", how="inner")
                    return combined.sort_values("date", ascending=False)
                else:
                    st.error("미세먼지 데이터 파일(CSV)을 찾을 수 없습니다.")
                    return pd.DataFrame()
            else:
                st.error(f"API 응답 컬럼이 부족합니다. 수신된 컬럼: {list(weather_df.columns)}")
                return pd.DataFrame()
        else:
            st.error("기상청 API 응답 형식이 올바르지 않거나 호출 한도를 초과했을 수 있습니다.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# 데이터 호출
df_env = get_latest_environment_data()

# ============================================================
# 2. 화면 구성 및 예측 로직
# ============================================================
if not df_env.empty:
    latest_row = df_env.iloc[0]
    st.success(f"✅ 데이터 로드 완료: {latest_row['date'].strftime('%Y-%m-%d')} 기준")

    # 대시보드 출력
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ 온도", f"{latest_row['temp']}°C")
    m2.metric("💧 습도", f"{latest_row['humidity']}%")
    m3.metric("☔ 강수", f"{latest_row['rainfall']}mm")
    m4.metric("💨 풍속", f"{latest_row['wind']}m/s")

    # (이후 모델 학습 및 예측 코드는 이전과 동일하게 작성)
    # ... [생략: RandomForest 학습 및 predict_proba 부분] ...
    
    st.info("💡 위 지표는 기상청 ASOS 관측소의 실시간 데이터를 바탕으로 합니다.")

else:
    st.warning("현재 분석 가능한 데이터를 가져올 수 없습니다. API 키 상태 또는 인터넷 연결을 확인해 주세요.")
