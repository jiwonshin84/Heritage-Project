# app.py
# ---------------------------------------
# 영천 문화유산 AI 분석 플랫폼 (Gemini API 내장판)
# Streamlit Cloud 배포용
# ---------------------------------------

import streamlit as st
import pandas as pd
import folium
import plotly.express as px
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import google.generativeai as genai

# ---------------------------------------
# Gemini API KEY 직접 입력
# ---------------------------------------
GEMINI_API_KEY = "AIzaSyCeNS_TTBIU6LmchWVdpki-Z9k0-MbKL6E"

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------
# 페이지 설정
# ---------------------------------------
st.set_page_config(
    page_title="영천 문화유산 AI 플랫폼",
    page_icon="🏛️",
    layout="wide"
)

# ---------------------------------------
# CSS 꾸미기
# ---------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1, h2, h3 {
    color:#7B241C;
}
.stButton>button {
    background:#7B241C;
    color:white;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------
# 제목
# ---------------------------------------
st.title("🏛️ 영천 문화유산 AI 분석 플랫폼")
st.caption("문화재청 OPEN API + AI 군집분석 + Gemini 설명형 AI")

# ---------------------------------------
# 데이터 로드
# ---------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("영천AI분석.csv")

df = load_data()

# ---------------------------------------
# 메뉴
# ---------------------------------------
menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "홈",
        "문화재 현황",
        "문화재 지도",
        "HeatMap",
        "AI 군집분석",
        "문화재 검색",
        "Gemini 챗봇"
    ]
)

# ---------------------------------------
# 홈
# ---------------------------------------
if menu == "홈":

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 문화재 수", len(df))
    col2.metric("국가유산 종목 수", df["국가유산종목"].nunique())
    col3.metric("분석 지역", "영천시")

    st.markdown("---")

    st.subheader("📌 프로젝트 소개")

    st.write("""
    본 플랫폼은 문화재청 OPEN API 데이터를 활용하여  
    영천 지역 문화유산을 분석하고 지도 시각화 및 AI 군집분석을 수행합니다.

    Gemini AI를 활용하여 문화재 설명과 정책 제안까지 제공합니다.
    """)

# ---------------------------------------
# 문화재 현황
# ---------------------------------------
elif menu == "문화재 현황":

    count = df["국가유산종목"].value_counts()

    fig = px.bar(
        x=count.values,
        y=count.index,
        orientation="h",
        title="영천 문화재 종목별 현황",
        labels={"x":"개수", "y":"종목"},
        color=count.values,
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# 지도
# ---------------------------------------
elif menu == "문화재 지도":

    m = folium.Map(
        location=[df["위도"].mean(), df["경도"].mean()],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    for _, row in df.iterrows():

        popup = f"""
        <b>{row['문화재명']}</b><br>
        종목: {row['국가유산종목']}<br>
        시대점수: {row['시대점수']}
        """

        folium.Marker(
            [row["위도"], row["경도"]],
            popup=popup,
            tooltip=row["문화재명"]
        ).add_to(m)

    st_folium(m, width=1300, height=700)

# ---------------------------------------
# HeatMap
# ---------------------------------------
elif menu == "HeatMap":

    m = folium.Map(
        location=[df["위도"].mean(), df["경도"].mean()],
        zoom_start=11
    )

    heat_data = df[["위도","경도"]].values.tolist()

    HeatMap(
        heat_data,
        radius=15
    ).add_to(m)

    st_folium(m, width=1300, height=700)

# ---------------------------------------
# AI 군집분석
# ---------------------------------------
elif menu == "AI 군집분석":

    X = df[["위도","경도","가치점수","시대점수"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(X_scaled)

    fig = px.scatter_mapbox(
        df,
        lat="위도",
        lon="경도",
        color="cluster",
        hover_name="문화재명",
        zoom=10,
        height=700
    )

    fig.update_layout(
        mapbox_style="open-street-map"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# 검색
# ---------------------------------------
elif menu == "문화재 검색":

    keyword = st.text_input("문화재명 검색")

    if keyword:
        result = df[df["문화재명"].str.contains(keyword, na=False)]
        st.dataframe(result)

# ---------------------------------------
# Gemini 챗봇
# ---------------------------------------
elif menu == "Gemini 챗봇":

    st.subheader("🤖 문화재 AI 챗봇")

    question = st.text_area(
        "질문 입력",
        placeholder="예: 은해사가 왜 문화재 밀집지역인가요?"
    )

    if st.button("질문하기"):

        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        너는 문화재 데이터 분석 전문가야.

        영천 문화재 데이터가 있다.
        총 문화재 수: {len(df)}
        주요 종목: {', '.join(df['국가유산종목'].unique()[:10])}

        사용자 질문:
        {question}

        학생 발표 수준으로 쉽게 설명해줘.
        """

        response = model.generate_content(prompt)

        st.success(response.text)
