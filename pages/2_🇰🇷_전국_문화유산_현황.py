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
<h1 style="font-size:34px; margin-bottom:5px;">
🇰🇷 전국 문화유산 현황
</h1>
<div style="font-size:17px; color:#6b7280; margin-bottom:20px;">
국가유산 공공데이터를 활용한 전국 문화유산 분포 시각화
</div>
""", unsafe_allow_html=True)

st.divider()

# =================================================
# 데이터 불러오기 (try-except로 파일 부재 시 대응)
# =================================================
try:
    df = pd.read_csv("data/raw/all_heritage.csv")
    # 컬럼 공백 제거
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    st.stop()

# =================================================
# 문화유산 중요도 점수 생성
# =================================================
importance_map = {
    "국보": 5, "보물": 4, "사적": 4, "명승": 3, 
    "천연기념물": 3, "국가민속문화유산": 3, 
    "국가등록문화유산": 2, "시도유형문화유산": 2, 
    "시도기념물": 2, "문화유산자료": 1
}

df["중요도점수"] = df["국가유산종목"].map(importance_map).fillna(1)

# =================================================
# 주요 지표 (Metrics)
# =================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("전체 문화유산", f"{len(df):,}개")
with c2:
    st.metric("문화유산 종목 수", f"{df['국가유산종목'].nunique()}종")
with c3:
    st.metric("시도 수", f"{df['시도명'].nunique()}개")
with c4:
    st.metric("가장 많은 지역", df["시도명"].value_counts().idxmax())

st.markdown("<br>", unsafe_allow_html=True)

# =================================================
# 1행: Treemap & Bubble Chart
# =================================================
row1_left, row1_right = st.columns([1.2, 1])

with row1_left:
    st.markdown("### 🗺 시도별 문화유산 분포")
    region_count = df["시도명"].value_counts().reset_index()
    region_count.columns = ["시도명", "개수"]

    fig1 = px.treemap(
        region_count, path=["시도명"], values="개수",
        color="개수", color_continuous_scale="YlOrRd"
    )
    fig1.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=500)
    fig1.update_traces(texttemplate="<b>%{label}</b><br>%{value}개", textfont_size=16)
    st.plotly_chart(fig1, use_container_width=True)

with row1_right:
    st.markdown("### 🫧 국가유산 종목별 현황")
    type_count = df["국가유산종목"].value_counts().reset_index()
    type_count.columns = ["국가유산종목", "개수"]

    n = len(type_count)
    cols = 4
    type_count["x"] = [i % cols for i in range(n)]
    type_count["y"] = [-(i // cols) for i in range(n)]

    fig2 = px.scatter(
        type_count, x="x", y="y", size="개수", color="개수",
        text="국가유산종목", size_max=60, color_continuous_scale="Blues"
    )
    fig2.update_traces(textposition="middle center", marker=dict(opacity=0.85, line=dict(width=1, color="white")))
    fig2.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), 
                      margin=dict(t=20, l=10, r=10, b=10), height=500, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# 2행: 경북 지역 상세 (Polar Bar & Heatmap)
# =================================================
st.markdown("<br>", unsafe_allow_html=True)
row2_left, row2_right = st.columns([1, 1])

# 경북 공통 데이터 추출
gb_df = df[df["시도명"] == "경북"].copy()

with row2_left:
    st.markdown("### 🌀 경북 지역 문화유산 분포 (상위 15개)")
    city_count = gb_df["시군구명"].value_counts().head(15).reset_index()
    city_count.columns = ["시군구명", "개수"]

    fig3 = px.bar_polar(
        city_count, r="개수", theta="시군구명",
        color="개수", color_continuous_scale="Tealgrn"
    )
    fig3.update_layout(height=500, margin=dict(t=50, b=50), coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

with row2_right:
    st.markdown("### 🌡 경북 시군구별 종목 현황")
    heatmap_df = pd.pivot_table(gb_df, index="시군구명", columns="국가유산종목", aggfunc="size", fill_value=0)
    top_cities = gb_df["시군구명"].value_counts().head(15).index
    heatmap_df = heatmap_df.loc[top_cities]

    fig4 = px.imshow(heatmap_df, text_auto=True, color_continuous_scale="YlGnBu", aspect="auto")
    fig4.update_layout(height=500, margin=dict(t=20, l=10, r=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

# =================================================
# 3행: 순위 & 비율 (Bar Charts)
# =================================================
st.markdown("<br>", unsafe_allow_html=True)
row3_left, row3_right = st.columns(2)

with row3_right:
    st.markdown("### 🏆 경북 지역 문화유산 순위")
    gb_rank = gb_df["시군구명"].value_counts().head(15).reset_index()
    gb_rank.columns = ["시군구명", "개수"]
    
    fig5 = px.bar(gb_rank.sort_values("개수"), x="개수", y="시군구명", orientation="h",
                 text="개수", color="개수", color_continuous_scale="Tealgrn")
    fig5.update_traces(textposition="outside")
    fig5.update_layout(height=500, margin=dict(t=20, l=10, r=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

with row3_left:
    st.markdown("### 🏺 지역별 국보 · 보물 비율")
    # 비율 계산
    total_count = gb_df.groupby("시군구명").size().reset_index(name="전체개수")
    treasure_count = gb_df[gb_df["국가유산종목"].isin(["국보", "보물"])].groupby("시군구명").size().reset_index(name="국보보물개수")
    
    ratio_df = pd.merge(total_count, treasure_count, on="시군구명", how="left").fillna(0)
    ratio_df["비율"] = (ratio_df["국보보물개수"] / ratio_df["전체개수"]) * 100
    ratio_df = ratio_df.sort_values("비율", ascending=False).head(15)

    fig6 = px.bar(ratio_df.sort_values("비율"), x="비율", y="시군구명", orientation="h",
                 color="비율", color_continuous_scale="Tealgrn")
    # 텍스트 레이블 수동 포맷팅
    fig6.update_traces(texttemplate='%{x:.1f}%', textposition="outside")
    fig6.update_layout(height=500, margin=dict(t=20, l=10, r=10, b=10), 
                      xaxis_title="비율 (%)", coloraxis_showscale=False)
    st.plotly_chart(fig6, use_container_width=True)

# =================================================
# 하단 설명
# =================================================
st.divider()
st.info("""
📌 **Treemap**: 지역별 문화유산 규모를 면적으로 표현합니다.  
📌 **Bubble Chart**: 국가유산 종목별 규모를 직관적으로 비교합니다.  
📌 **Polar Bar**: 방사형 구조를 통해 지역별 분포의 균형을 시각화합니다.  
📌 **Heatmap**: 지역과 종목 간의 밀집도를 한눈에 파악할 수 있습니다.
""")
