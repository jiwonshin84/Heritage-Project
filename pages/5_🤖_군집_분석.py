import streamlit as st
import pandas as pd
import plotly.express as px

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="영천 국가유산 군집분석 리포트",
    layout="wide"
)

st.title("🤖 국가유산 지능형 군집분석")
st.markdown("데이터 분석을 통해 영천시 국가유산들의 특성별 그룹을 식별합니다.")
st.divider()

# =================================================
# 데이터 로드
# =================================================
@st.cache_data
def load_cluster_data():
    df = pd.read_csv("data/processed/yc_clustering.csv")
    # cluster 컬럼을 문자열로 변환하여 이산형 색상 적용
    df["cluster"] = df["cluster"].astype(str)
    return df

df = load_cluster_data()

# =================================================
# 상단 통계 요약 (Metrics)
# =================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("전체 유산 수", f"{len(df)}건")
with c2:
    st.metric("식별된 군집 수", f"{df['cluster'].nunique()}개")
with c3:
    st.metric("평균 가치점수", f"{df['가치점수'].mean():.2f}")
with c4:
    st.metric("평균 시대점수", f"{df['시대점수'].mean():.2f}")

st.divider()

# =================================================
# 메인 분석 레이아웃
# =================================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📍 공간적 군집 분포")
    st.caption("위경도 좌표를 바탕으로 유산들이 어디에 밀집되어 있는지 확인합니다.")
    
    # 산점도 개선 (크기를 가치점수에 연동)
    fig_scatter = px.scatter(
        df,
        x="경도",
        y="위도",
        color="cluster",
        size="가치점수",  # 점수가 높을수록 마커가 커짐
        hover_data=["문화재명(국문)", "국가유산종목", "시대"],
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_white"
    )
    fig_scatter.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📊 군집별 특성 비교")
    st.caption("각 군집이 어떤 점수대(가치/시대)에 집중되어 있는지 비교합니다.")

    # 군집별 평균 계산
    summary = df.groupby("cluster")[["가치점수", "시대점수"]].mean().reset_index()
    
    # 막대 그래프 시각화 추가
    fig_bar = px.bar(
        summary,
        x="cluster",
        y=["가치점수", "시대점수"],
        barmode="group",
        labels={"value": "평균 점수", "variable": "항목"},
        color_discrete_map={"가치점수": "#636EFA", "시대점수": "#EF553B"},
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 데이터프레임 표시 (스타일링 적용)
    st.dataframe(
        summary.style.highlight_max(axis=0, color='#e6f3ff'),
        use_container_width=True
    )

# =================================================
# 군집별 상세 데이터 탐색기
# =================================================
st.divider()
st.subheader("🔍 군집별 상세 데이터 확인")
selected_cluster = st.selectbox("확인할 군집을 선택하세요", sorted(df["cluster"].unique()))

cluster_details = df[df["cluster"] == selected_cluster][
    ["문화재명(국문)", "국가유산종목", "시대", "가치점수", "시대점수"]
]
st.dataframe(cluster_details, use_container_width=True)
