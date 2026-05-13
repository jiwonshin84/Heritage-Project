import streamlit as st
import pandas as pd
import google.generativeai as genai

# =====================================================
# 1. 페이지 설정
# =====================================================
st.set_page_config(
    page_title="영천 AI 문화재 해설사",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# 2. Gemini API 설정
# =====================================================
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    st.error("API 키가 설정되지 않았습니다. .streamlit/secrets.toml을 확인하세요.")

# =====================================================
# 3. 데이터 로드 및 전처리
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/yc_heritage_detail_enriched.csv")

def clean(val):
    if pd.isna(val) or str(val).strip() == "":
        return "-"
    return str(val).strip()

# =====================================================
# 4. 메인 UI 로직
# =====================================================
try:
    df = load_data()
    
    st.title("🤖 AI 문화재 해설 가이드")
    st.markdown("---")

    # 선택 영역
    category_col = "종목" if "종목" in df.columns else "국가유산종목"
    col_sel1, col_sel2 = st.columns(2)
    
    with col_sel1:
        category = st.selectbox("📂 문화재 품목 선택", sorted(df[category_col].dropna().unique()))
    
    filtered_df = df[df[category_col] == category]
    
    with col_sel2:
        heritage = st.selectbox("🏛 문화재 선택", filtered_df["문화재명(국문)"])

    row = filtered_df[filtered_df["문화재명(국문)"] == heritage].iloc[0]

    # 상단 제목 배너
    st.markdown(f"""
        <div style="background-color:#f8f9fa; padding:20px; border-radius:15px; text-align:center; margin-bottom:25px; border:1px solid #e9ecef;">
            <h1 style="margin:0; color:#2c3e50; font-size:32px;">🏛 {heritage}</h1>
        </div>
    """, unsafe_allow_html=True)

    # 콘텐츠 영역 (이미지 1 : 정보 1.2)
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        image_url = row.get("이미지URL")
        if pd.notna(image_url) and str(image_url).strip() != "":
            st.image(image_url, use_container_width=True)
            st.caption(f"출처: 국가유산청 - {heritage}")
        else:
            st.info("🖼 등록된 이미지가 없습니다.")

    with right_col:
        st.markdown("<h3 style='margin-top:0; color:#2c3e50;'>📋 상세 정보</h3>", unsafe_allow_html=True)
        
        # 정보 데이터 정리
        info_data = {
            "종목": clean(row.get(category_col)),
            "분류": f"{clean(row.get('국가유산분류'))} ({clean(row.get('국가유산분류2'))})",
            "한자명": clean(row.get("문화재명(한자)")),
            "시대": clean(row.get("시대")),
            "소재지": clean(row.get("소재지상세")),
            "소유/관리": f"{clean(row.get('소유자'))} / {clean(row.get('관리자'))}"
        }

        # ⭐️ 중요: HTML 테이블 문자열 생성
        # 반드시 <table> 태그부터 시작해야 합니다.
        info_html = """
        <style>
            .custom-table { width: 100%; border-collapse: collapse; margin-top: 10px; border-radius: 10px; overflow: hidden; }
            .custom-tr { border-bottom: 1px solid #eeeeee; }
            .custom-td-key { 
                width: 25%; padding: 12px 10px; font-weight: bold; color: #34495e; 
                background-color: #fcfcfc; font-size: 15px;
            }
            .custom-td-val { 
                width: 75%; padding: 12px 15px; color: #2c3e50; 
                font-size: 15px; line-height: 1.5;
            }
        </style>
        <table class="custom-table">
        """
        for key, val in info_data.items():
            info_html += f"""
            <tr class="custom-tr">
                <td class="custom-td-key">{key}</td>
                <td class="custom-td-val">{val}</td>
            </tr>
            """
        info_html += "</table>"
        
        # 🔥 이 부분이 핵심입니다! unsafe_allow_html=True가 없으면 코드가 그대로 보입니다.
        st.markdown(info_html, unsafe_allow_html=True)
        
        with st.expander("📖 원문 설명 보기", expanded=True):
            st.write(clean(row.get("내용")))

    # AI 기능 섹션
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.header("🤖 AI 스마트 도슨트")
    
    col_ai1, col_ai2 = st.columns(2)
    with col_ai1:
        if st.button("✨ AI 도슨트 해설 생성"):
            with st.spinner("해설을 작성 중입니다..."):
                prompt = f"{heritage}에 대해 도슨트처럼 설명해줘. 내용: {clean(row.get('내용'))}"
                response = model.generate_content(prompt)
                st.info(response.text)
                
    with col_ai2:
        user_q = st.text_input("💬 궁금한 점을 질문해보세요")
        if st.button("질문 전송"):
            if user_q:
                with st.spinner("답변 중..."):
                    res = model.generate_content(f"{heritage} 관련 질문: {user_q}. 설명: {clean(row.get('내용'))}")
                    st.success(res.text)

except Exception as e:
    st.error(f"오류 발생: {e}")
