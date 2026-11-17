import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="영화 관람 습관 분석", page_icon="🎬", layout="wide")
st.title("🎬 영화 관람 습관 분석")
st.write("영화 장르, 관람 빈도, 평점 등을 분석하고 선호 장르를 예측합니다.")
st.caption("※ 실제 설문 결과가 아닌 프로그램 내부의 가상 예시 데이터 300개입니다.")

@st.cache_data
def make_data():
    rng = np.random.default_rng(24)
    moods = ["즐거움", "긴장감", "감동", "편안함"]
    places = ["집", "영화관"]
    genres = ["액션", "코미디", "로맨스", "판타지", "스릴러", "애니메이션"]
    rows = []
    for _ in range(300):
        mood = rng.choice(moods)
        place = rng.choice(places)
        frequency = int(rng.integers(1, 9))
        rating = round(float(rng.uniform(2, 5)), 1)
        if mood == "긴장감":
            genre = rng.choice(["액션", "스릴러"])
        elif mood == "감동":
            genre = rng.choice(["로맨스", "애니메이션"])
        elif mood == "편안함":
            genre = rng.choice(["코미디", "판타지", "애니메이션"])
        else:
            genre = rng.choice(genres)
        rows.append([mood, place, frequency, rating, genre])
    return pd.DataFrame(rows, columns=["관람할 때 기분", "주요 관람 장소", "월 관람 횟수", "평점", "선호 장르"])

df = make_data()
a, b, c = st.columns(3)
a.metric("분석 데이터", f"{len(df)}개")
b.metric("평균 영화 평점", f"{df['평점'].mean():.1f} / 5.0")
c.metric("가장 인기 있는 장르", df["선호 장르"].mode()[0])

st.subheader("1. 영화 관람 예시 데이터")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("2. 영화 취향 그래프")
left, right = st.columns(2)
with left:
    st.write("**장르별 선호도**")
    fig, ax = plt.subplots()
    df["선호 장르"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("영화 장르")
    ax.set_ylabel("선호 인원")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.write("**장르별 평균 평점**")
    fig, ax = plt.subplots()
    df.groupby("선호 장르")["평점"].mean().sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_xlabel("영화 장르")
    ax.set_ylabel("평균 평점")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("3. 머신러닝 모델 정확도 비교")
X = pd.get_dummies(df[["관람할 때 기분", "주요 관람 장소"]])
X["월 관람 횟수"] = df["월 관람 횟수"]
X["평점"] = df["평점"]
y = df["선호 장르"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
for model in models.values():
    model.fit(X_train, y_train)
scores = [accuracy_score(y_test, model.predict(X_test)) for model in models.values()]
st.dataframe(pd.DataFrame({"모델": list(models.keys()), "정확도": [f"{s:.1%}" for s in scores]}),
             hide_index=True, use_container_width=True)

st.subheader("4. 나의 영화 취향 예측")
col1, col2 = st.columns(2)
with col1:
    mood = st.selectbox("영화를 볼 때 원하는 기분", ["즐거움", "긴장감", "감동", "편안함"])
    place = st.selectbox("주요 관람 장소", ["집", "영화관"])
with col2:
    frequency = st.slider("한 달 평균 관람 횟수", 1, 15, 3)
    rating = st.slider("평소 영화 평점", 1.0, 5.0, 4.0, 0.5)

if st.button("내 선호 장르 예측하기"):
    sample = pd.DataFrame([[mood, place, frequency, rating]],
                          columns=["관람할 때 기분", "주요 관람 장소", "월 관람 횟수", "평점"])
    sample_X = pd.get_dummies(sample[["관람할 때 기분", "주요 관람 장소"]])
    sample_X["월 관람 횟수"] = sample["월 관람 횟수"]
    sample_X["평점"] = sample["평점"]
    sample_X = sample_X.reindex(columns=X.columns, fill_value=0)
    result = models["Random Forest"].predict(sample_X)[0]
    st.success(f"예측된 선호 영화 장르는 **{result}** 입니다.")

st.info("CSV 파일을 사용하지 않으므로 CSV 열 이름 오류가 발생하지 않습니다.")
