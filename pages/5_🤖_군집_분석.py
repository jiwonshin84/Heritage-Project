import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="영천 국가유산 군집분석 리포트",
    layout="wide"
)

st.title("🤖 국가유산 지능형 군집분석 시스템")
st.markdown("영천시 국가유산 105건의 **가치/시대/위치** 데이터를 바탕으로 실시간 군집화를 수행합니다.")
st.divider()

# =================================================
# 데이터 로드 및 전처리
# =================================================
@st.cache_data
def load_base_data():
    # 가치점수와 시대점수가 포함된 데이터를 로드합니다.
    df = pd.read_csv("data/processed/yc_clustering.csv")
    df = df.dropna(subset=['위도', '경도', '가치점수', '시대점수'])
    return df

df_base = load_base_data()

# =================================================
# 사이드바 설정 (실시간 분석 엔진 제어)
# =================================================
st.sidebar.header("⚙️ 분석 엔진 설정")
k_value = st.sidebar.slider("군집 수(k) 설정", min_value=2, max_value=10, value=3)

# =================================================
# 실시간 K-Means 분석 수행
# =================================================
features = df_base[['위도', '경도', '가치점수', '시대점수']]
kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42, n_init=10)
df_base['cluster'] = kmeans.fit_predict(features).astype(str)

# 실시간 실루엣 계수 계산
sil_avg = silhouette_score(features, df_base['cluster'])
df_base['silhouette'] = silhouette_samples(features, df_base['cluster'])

# =================================================
# 상단 요약 지표 (Metrics)
# =================================================
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("분석 대상 전수", "105건")
with c2: st.metric("현재 설정된 k", f"{k_value}개")
with c3: st.metric("평균 가치점수", f"{df_base['가치점수'].mean():.2f}")
with c4: st.metric("실시간 실루엣 계수", f"{sil_avg:.3f}")

st.divider()

# =================================================
# 메인 분석 시각화 (공간 및 지표)
# =================================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader(f"📍 공간적 군집 분포 (k={k_value})")
    fig_scatter = px.scatter(
        df_base, x="경도", y="위도", color="cluster", size="가치점수",
        hover_data=["문화재명(국문)", "국가유산종목"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white",
        category_orders={"cluster": [str(i) for i in range(k_value)]}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("📊 군집별 평균 지표")
    summary = df_base.groupby("cluster").agg({
        "가치점수": "mean", "시대점수": "mean", "silhouette": "mean", "문화재명(국문)": "count"
    }).rename(columns={"문화재명(국문)": "개수"}).reset_index()
    
    fig_bar = px.bar(
        summary, x="cluster", y=["가치점수", "시대점수"], barmode="group",
        template="plotly_white", color_discrete_map={"가치점수": "#636EFA", "시대점수": "#EF553B"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# =================================================
# 하단: 군집별 필터링 상세 목록 (핵심 추가 부분)
# =================================================
st.divider()
st.subheader("🔍 군집별 상세 목록 탐색")
st.markdown("각 군집 탭을 클릭하여 해당 그룹에 속한 국가유산 리스트를 확인하세요.")

# 군집 번호순으로 정렬하여 탭 생성
cluster_labels = sorted(df_base['cluster'].unique(), key=int)
tabs = st.tabs([f"군집 {c}" for c in cluster_labels])

for i, tab in enumerate(tabs):
    cluster_id = cluster_labels[i]
    with tab:
        # 해당 군집 데이터만 필터링
        cluster_df = df_base[df_base['cluster'] == cluster_id].sort_values("가치점수", ascending=False)
        
        # 군집 요약 정보 요약
        cnt = len(cluster_df)
        avg_v = cluster_df['가치점수'].mean()
        st.write(f"✅ 이 군집에는 **{cnt}개**의 유산이 포함되어 있으며, 평균 가치점수는 **{avg_v:.2f}**입니다.")
        
        # 데이터프레임 출력
        st.dataframe(
            cluster_df[["문화재명(국문)", "국가유산종목", "시대", "가치점수", "시대점수", "silhouette"]],
            use_container_width=True,
            hide_index=True
        )

# =================================================
# 하단 가이드
# =================================================
st.info("""
**💡 분석 팁:** 하단 목록에서 특정 군집의 **가치점수**가 유독 높다면 해당 군집은 영천시의 핵심 유산 관리 그룹으로 볼 수 있습니다. 
반면 **실루엣 계수**가 낮은 항목이 섞여 있다면, 그 유산은 다른 군집으로 이동할 가능성이 있는 경계 지점의 데이터임을 의미합니다.
""")
