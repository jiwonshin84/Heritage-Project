import streamlit as st
import pandas as pd
import folium
import re
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster

# =================================================
# 페이지 설정
# =================================================
st.set_page_config(
    page_title="영천 국가유산 공간 정보",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 스타일 (목록 버튼 디자인 정돈)
st.markdown("""
    <style>
    .stButton>button {
        text-align: left;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: -10px;
    }
    hr {
        margin: 10px 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================
# 제목 및 헤더
# =================================================
st.title("🛰️ 영천 국가유산 공간 정보")
st.markdown("##### 영천시 내 국가유산의 위치와 시대별 분포를 시각적으로 분석합니다.")
st.divider()

# =================================================
# 데이터 로드 및 전처리 (캐싱)
# =================================================
@st.cache_data
def load_data():
    # 파일 경로를 사용자 환경에 맞게 조정하세요.
    try:
        df = pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")
    except FileNotFoundError:
        # 파일이 없을 경우를 위한 빈 데이터프레임 예시
        return pd.DataFrame(columns=["문화재명(국문)", "위도", "경도", "시대", "국가유산종목", "소재지상세", "이미지URL"])
    
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["위도", "경도"])
    
    def simplify_era(text):
        if pd.isna(text): return "기타"
        text = str(text).strip()
        if "청동기" in text: return "청동기"
        elif any(x in text for x in ["통일신라", "신라시대 후기"]): return "통일신라"
        elif "신라" in text: return "신라"
        elif "고려" in text:
            if any(x in text for x in ["초기", "전기"]): return "고려초기"
            if any(x in text for x in ["말기", "후기"]): return "고려후기"
            return "고려"
        elif any(k in text for k in ["세종", "태조", "태종", "문종", "단종", "세조", "성종", "연산군", "중종", "인종"]):
            return "조선초기"
        elif any(k in text for k in ["숙종", "영조", "정조", "순조", "철종", "고종", "광해군"]):
            return "조선후기"
        elif "조선" in text:
            if any(x in text for x in ["초기", "전기"]): return "조선초기"
            if any(x in text for x in ["말기", "후기"]): return "조선후기"
            return "조선"
        elif "대한제국" in text: return "대한제국"
        
        year_match = re.search(r"\d{4}", text)
        if year_match:
            yr = int(year_match.group())
            if yr < 700: return "신라"
            elif yr < 1400: return "고려"
            elif yr < 1600: return "조선초기"
            elif yr < 1910: return "조선후기"
        return "기타"

    df["시대그룹"] = df["시대"].apply(simplify_era)
    df["국가유산종목"] = df.get("국가유산종목", "미상").fillna("미상").astype(str)
    df["소재지상세"] = df.get("소재지상세", "-").fillna("-").astype(str)
    return df

df = load_data()

# =================================================
# 사이드바 필터
# =================================================
st.sidebar.header("🔎 검색 및 필터")

era_order = ["청동기", "신라", "통일신라", "고려초기", "고려", "고려후기", "조선초기", "조선", "조선후기", "대한제국", "기타"]
existing_eras = [e for e in era_order if e in df["시대그룹"].unique()]
selected_era = st.sidebar.selectbox("시대", ["전체"] + existing_eras)

type_options = ["전체"] + sorted(df["국가유산종목"].unique().tolist())
selected_type = st.sidebar.selectbox("종목", type_options)

# 필터 적용
filtered_df = df.copy()
if selected_era != "전체":
    filtered_df = filtered_df[filtered_df["시대그룹"] == selected_era]
if selected_type != "전체":
    filtered_df = filtered_df[filtered_df["국가유산종목"] == selected_type]

st.sidebar.metric("검색 결과", f"{len(filtered_df)} 개")

if filtered_df.empty:
    st.warning("선택한 조건에 맞는 유산이 없습니다.")
    st.stop()

# =================================================
# 세션 상태 (선택 항목 관리)
# =================================================
if "selected_heritage" not in st.session_state or \
   st.session_state.selected_heritage not in filtered_df["문화재명(국문)"].values:
    st.session_state.selected_heritage = filtered_df.iloc[0]["문화재명(국문)"]

# 현재 선택된 데이터
selected_row = filtered_df[filtered_df["문화재명(국문)"] == st.session_state.selected_heritage].iloc[0]

# =================================================
# 레이아웃: 지도(Left) + 목록(Right)
# =================================================
map_col, list_col = st.columns([3.5, 1.2])

with map_col:
    # 1. 지도 기본 설정
    m = folium.Map(
        location=[selected_row["위도"], selected_row["경도"]],
        zoom_start=14,
        tiles="CartoDB positron"
    )

    # 2. 클러스터 및 히트맵 추가
    marker_cluster = MarkerCluster().add_to(m)
    heat_data = filtered_df[["위도", "경도"]].values.tolist()
    HeatMap(heat_data, radius=20, blur=15).add_to(m)

    # 3. 개별 마커 생성
    for _, row in filtered_df.iterrows():
        name = row["문화재명(국문)"]
        is_selected = (name == st.session_state.selected_heritage)
        
        # 팝업 이미지 HTML
        img_url = str(row.get("이미지URL", "")).replace("http://", "https://")
        if not img_url or img_url.lower() == "nan":
            img_tag = '<div style="width:100%;height:150px;background:#f0f0f0;display:flex;justify-content:center;align-items:center;border-radius:10px;">이미지 없음</div>'
        else:
            img_tag = f'<img src="{img_url}" style="width:100%;height:180px;object-fit:cover;border-radius:10px;">'

        popup_html = f"""
            <div style="width:280px; font-family: 'Malgun Gothic', sans-serif;">
                <h4 style="margin-bottom:10px;">{name}</h4>
                {img_tag}
                <table style="width:100%; font-size:12px; margin-top:10px; border-collapse:collapse;">
                    <tr><td style="padding:4px; font-weight:bold; width:40px;">시대</td><td>{row['시대그룹']}</td></tr>
                    <tr><td style="padding:4px; font-weight:bold;">종목</td><td>{row['국가유산종목']}</td></tr>
                    <tr><td style="padding:4px; font-weight:bold;">주소</td><td>{row['소재지상세']}</td></tr>
                </table>
            </div>
        """
        
        folium.Marker(
            location=[row["위도"], row["경도"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=name,
            icon=folium.Icon(color="red" if is_selected else "blue", icon="university", prefix="fa")
        ).add_to(marker_cluster)

    # 지도 출력
    st_folium(m, width="100%", height=750, key="heritage_gis_map")

with list_col:
    st.subheader("🏛️ 국가유산 목록")
    st.caption("항목을 클릭하면 지도가 이동합니다.")
    
    # 목록 영역 (스크롤 가능 컨테이너)
    with st.container(height=650):
        for idx, row in filtered_df.iterrows():
            name = row["문화재명(국문)"]
            addr = row["소재지상세"]
            is_selected = (name == st.session_state.selected_heritage)
            
            # 버튼 클릭 이벤트
            # 선택된 상태면 버튼의 색상이나 포맷을 다르게 표시할 수 있습니다.
            btn_label = f"📍 {name}" if is_selected else f"🏛️ {name}"
            
            if st.button(btn_label, key=f"list_btn_{idx}", use_container_width=True):
                st.session_state.selected_heritage = name
                st.rerun()
            
            # 주소 상세 정보 표시 (HTML 코드 노출 방지를 위해 텍스트로 처리)
            st.caption(f"위치: {addr}")
            st.divider()
