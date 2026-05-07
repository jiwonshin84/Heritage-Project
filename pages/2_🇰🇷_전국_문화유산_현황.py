import streamlit as st
import pandas as pd
import plotly.express as px

# =================================================
# 페이지 설정
# =================================================

st.set_page_config(
    page_title="전국 문화유산 현황",
    page_icon="🏛",
    layout="wide"
)

# =================================================
# 제목
# =================================================

st.markdown("""
<h1 style="
font-size:34px;
margin-bottom:5px;
">
🇰🇷 전국 문화유산 현황
</h1>

<div style="
font-size:17px;
color:#6b7280;
margin-bottom:20px;
">
국가유산 공공데이터를 활용한 전국 문화유산 분포 시각화
</div>
""", unsafe_allow_html=True)

st.divider()

# =================================================
# 데이터 불러오기
# =================================================

df = pd.read_csv(
    "data/raw/all_heritage.csv"
)

# 컬럼 공백 제거
df.columns = df.columns.str.strip()

# =================================================
# 주요 지표
# =================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "전체 문화유산",
        f"{len(df):,}개"
    )

with c2:

    st.metric(
        "문화유산 종목 수",
        df["국가유산종목"].nunique()
    )

with c3:

    st.metric(
        "시도 수",
        df["시도명"].nunique()
    )

with c4:

    st.metric(
        "가장 많은 지역",
        df["시도명"].value_counts().idxmax()
    )

st.markdown("<br>", unsafe_allow_html=True)

# =================================================
# 레이아웃
# =================================================

left, right = st.columns([1.2, 1])

# =================================================
# 1. Treemap
# =================================================

with left:

    st.markdown("""
    <h3 style="
    font-size:24px;
    margin-bottom:10px;
    ">
    🗺 시도별 문화유산 분포
    </h3>
    """, unsafe_allow_html=True)

    region_count = (
        df["시도명"]
        .value_counts()
        .reset_index()
    )

    region_count.columns = [
        "시도명",
        "개수"
    ]

    fig1 = px.treemap(

        region_count,

        path=["시도명"],

        values="개수",

        color="개수",

        color_continuous_scale="YlOrRd"

    )

    fig1.update_layout(

        margin=dict(
            t=20,
            l=10,
            r=10,
            b=10
        ),

        height=650
    )

    fig1.update_traces(

        textfont_size=18,

        hovertemplate=
        "<b>%{label}</b><br>" +
        "문화유산 수: %{value}개"

    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# =================================================
# 2. Bubble Chart
# =================================================

with right:

    st.markdown("""
    <h3 style="
    font-size:24px;
    margin-bottom:10px;
    ">
    🫧 국가유산 종목별 현황
    </h3>
    """, unsafe_allow_html=True)

    type_count = (
        df["국가유산종목"]
        .value_counts()
        .reset_index()
    )

    type_count.columns = [
        "국가유산종목",
        "개수"
    ]

    # 위치용 숫자 생성
    type_count["x"] = [
        1, 2, 3, 4, 5,
        1.5, 2.5, 3.5, 4.5,
        2, 3, 4
    ][:len(type_count)]

    type_count["y"] = [
        1, 1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3
    ][:len(type_count)]

    fig2 = px.scatter(

        type_count,

        x="x",
        y="y",

        size="개수",

        color="개수",

        text="국가유산종목",

        size_max=120,

        color_continuous_scale="Blues"

    )

    fig2.update_traces(

        textposition="middle center",

        marker=dict(
            opacity=0.85,
            line=dict(
                width=2,
                color="white"
            )
        ),

        hovertemplate=
        "<b>%{text}</b><br>" +
        "문화유산 수: %{marker.size}개"
    )

    fig2.update_layout(

        xaxis=dict(
            visible=False
        ),

        yaxis=dict(
            visible=False
        ),

        showlegend=False,

        margin=dict(
            t=20,
            l=10,
            r=10,
            b=10
        ),

        height=650
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# =================================================
# 하단 설명
# =================================================

st.markdown("<br>", unsafe_allow_html=True)

st.info("""
📌 Treemap은 지역별 문화유산 규모를 면적으로 표현하며,  
Bubble Chart는 국가유산 종목별 분포를 직관적으로 보여줍니다.
""")
