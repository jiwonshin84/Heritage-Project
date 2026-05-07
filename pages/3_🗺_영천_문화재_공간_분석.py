import streamlit as st
import pandas as pd
import folium

from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

# =================================================
# 페이지 설정
# =================================================

st.set_page_config(
    page_title="영천 문화재 공간분석",
    layout="wide"
)

st.title("🗺 영천 문화재 공간분석")

# =================================================
# 데이터 불러오기
# =================================================

df = pd.read_csv(
    "data/processed/yc_heritage_detail_enriched.csv"
)

# 위경도 결측 제거
df = df.dropna(subset=["위도", "경도"])

# =================================================
# 사이드바 필터
# =================================================

st.sidebar.header("🔎 문화재 필터")

era_list = ["전체"] + sorted(
    df["시대"].dropna().unique().tolist()
)

selected_era = st.sidebar.selectbox(
    "시대 선택",
    era_list
)

if selected_era != "전체":
    df = df[df["시대"] == selected_era]

# =================================================
# 지도 생성
# =================================================

center = [
    df["위도"].mean(),
    df["경도"].mean()
]

m = folium.Map(
    location=center,
    zoom_start=10,
    tiles="CartoDB positron"
)

# =================================================
# 클러스터
# =================================================

marker_cluster = MarkerCluster().add_to(m)

# =================================================
# 마커 생성
# =================================================

for idx, row in df.iterrows():

    image_url = str(
        row.get("이미지URL", "")
    ).strip()

    heritage_name = row.get(
        "문화재명(국문)",
        "문화재"
    )

    popup_html = f"""
    <div style="
        width:260px;
        padding:10px;
    ">

        <h4 style="
            margin-bottom:10px;
            color:#222;
        ">
            {heritage_name}
        </h4>

        <a href="{image_url}" target="_blank">

            <img
                src="{image_url}"
                width="240"
                style="
                    border-radius:12px;
                    margin-bottom:10px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.2);
                "
            >

        </a>

        <table style="
            width:100%;
            font-size:14px;
        ">

            <tr>
                <td><b>시대</b></td>
                <td>{row.get("시대", "-")}</td>
            </tr>

            <tr>
                <td><b>종목</b></td>
                <td>{row.get("국가유산종목", "-")}</td>
            </tr>

            <tr>
                <td><b>재질</b></td>
                <td>{row.get("재질", "-")}</td>
            </tr>

            <tr>
                <td><b>노출형태</b></td>
                <td>{row.get("노출형태", "-")}</td>
            </tr>

        </table>

        <br>

        <div style="
            font-size:13px;
            color:#555;
            line-height:1.5;
        ">
            이미지를 클릭하면 크게 볼 수 있습니다.
        </div>

    </div>
    """

    folium.Marker(
        location=[
            row["위도"],
            row["경도"]
        ],

        popup=folium.Popup(
            popup_html,
            max_width=300
        ),

        tooltip=heritage_name,

        icon=folium.Icon(
            color="red",
            icon="info-sign"
        )

    ).add_to(marker_cluster)

# =================================================
# HeatMap
# =================================================

heat_data = df[
    ["위도", "경도"]
].values.tolist()

HeatMap(
    heat_data,
    radius=20,
    blur=15
).add_to(m)

# =================================================
# 지도 출력
# =================================================

st_data = st_folium(
    m,
    width=1400,
    height=750
)

# =================================================
# 하단 카드형 목록 UI
# =================================================

st.markdown("---")
st.subheader("🏛 문화재 상세 카드")

cols = st.columns(3)

for idx, row in df.head(9).iterrows():

    with cols[idx % 3]:

        st.markdown(f"""
        <div style="
            border-radius:18px;
            padding:15px;
            box-shadow:0 2px 12px rgba(0,0,0,0.1);
            margin-bottom:20px;
            background:white;
        ">
        """, unsafe_allow_html=True)

        st.image(
            row["이미지URL"],
            use_container_width=True
        )

        st.markdown(f"""
        ### {row["문화재명(국문)"]}
        """)
        
        st.markdown(f"""
        - **시대** : {row.get("시대", "-")}
        - **종목** : {row.get("국가유산종목", "-")}
        - **재질** : {row.get("재질", "-")}
        """)

        with st.expander("📖 상세 설명"):

            st.write(
                row.get("내용", "설명 없음")
            )

        st.markdown("</div>", unsafe_allow_html=True)
