import streamlit as st
import pandas as pd

st.title("⚠ 문화재 훼손 위험도 예측")

material = st.selectbox(
    "재질",
    ["목조", "석조", "금속", "벽화"]
)

humidity = st.slider(
    "습도",
    0,
    100,
    70
)

dust = st.slider(
    "미세먼지",
    0,
    200,
    50
)

outdoor = st.selectbox(
    "노출형태",
    ["실내", "실외"]
)

# 간단 위험도 계산
risk = 0

if material == "목조":
    risk += 30

risk += humidity * 0.3
risk += dust * 0.2

if outdoor == "실외":
    risk += 20

risk = min(100, int(risk))

st.metric(
    "예상 훼손 위험도",
    f"{risk}%"
)

if risk >= 80:
    st.error("매우 위험")
elif risk >= 60:
    st.warning("위험")
else:
    st.success("보통")
