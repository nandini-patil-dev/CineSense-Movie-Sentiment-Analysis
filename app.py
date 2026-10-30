

import streamlit as st
import joblib
from pathlib import Path
from preprocess import clean_text

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CineSense | AI Movie Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.joblib"

model = joblib.load(MODEL_PATH)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #090909, #171717, #240909);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(
        90deg,
        rgba(0,0,0,0.95),
        rgba(0,0,0,0.65),
        rgba(0,0,0,0.25)
    ),
    url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    background-size: cover;
    background-position: center;
    padding: 75px 45px;
    border-radius: 20px;
    border: 1px solid #553030;
    margin-bottom: 35px;
}

.hero h1 {
    font-size: 48px;
    color: white;
    font-weight: 900;
}

.hero p {
    font-size: 18px;
    color: #dddddd;
}

.tag {
    color: #ff4545;
    font-weight: bold;
    letter-spacing: 3px;
}

.section-title {
    font-size: 27px;
    font-weight: bold;
    color: white;
    margin-top: 30px;
    margin-bottom: 15px;
}

.poster-card {
    background: #1c1c1c;
    padding: 12px;
    border-radius: 15px;
    border: 1px solid #393939;
    text-align: center;
}

.poster-card h4 {
    color: white;
}

.poster-card p {
    color: #bbbbbb;
    font-size: 13px;
}

textarea {
    background-color: #171717 !important;
    color: white !important;
    border: 1px solid #aa2929 !important;
    border-radius: 12px !important;
}

.stButton > button {
    background: linear-gradient(90deg, #e50914, #a00000);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #ff3030;
    color: white;
    border: 1px solid white;
}

.result-card {
    background: #202020;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #663333;
    text-align: center;
}

.footer {
    text-align: center;
    color: #888888;
    padding: 35px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">
    <div class="tag">YOUR PERSONAL AI MOVIE CRITIC</div>
    <h1>Lights. Camera.<br>Sentiment!</h1>
    <p>
        Discover the emotions behind every movie review.
        <br>
        Let Artificial Intelligence understand what you feel.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- MOVIE GALLERY ----------------

st.markdown(
    '<div class="section-title">🍿 Explore the Cinematic World</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

movies = [
    (
        "Cinema Experience",
        "https://images.unsplash.com/photo-1440404653325-ab127d49abc1",
        "The magic of filmmaking"
    ),
    (
        "Movie Night",
        "https://images.unsplash.com/photo-1478720568477-152d9b164e26",
        "Stories that touch hearts"
    ),
    (
        "Film Moments",
        "https://images.unsplash.com/photo-1485846234645-a62644f84728",
        "Every frame tells a story"
    ),
    (
        "The Big Screen",
        "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c",
        "Experience the emotions"
    )
]

for col, movie in zip([col1, col2, col3, col4], movies):
    with col:
        st.image(movie[1], use_container_width=True)
        st.markdown(
            f"""
            <div class="poster-card">
                <h4>{movie[0]}</h4>
                <p>{movie[2]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------- REVIEW ANALYZER ----------------

st.markdown(
    '<div class="section-title">🎭 Analyze Your Movie Review</div>',
    unsafe_allow_html=True
)

st.write(
    "Write your thoughts about any movie. "
    "Our AI model will predict the sentiment."
)

review = st.text_area(
    "YOUR MOVIE REVIEW",
    placeholder="Example: The movie was absolutely amazing! "
                "The acting, storyline and music were fantastic...",
    height=170,
    key="review_input"
)

analyze = st.button("🎬 ANALYZE MY REVIEW")


# ---------------- PREDICTION ----------------

if analyze:

    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        cleaned_review = clean_text(review)

        prediction = model.predict([cleaned_review])[0]
        probabilities = model.predict_proba([cleaned_review])[0]

        confidence = probabilities[prediction] * 100

        st.markdown("---")
        st.markdown(
            '<div class="section-title">🎯 AI Analysis Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.markdown("""
            <div class="result-card">
                <h1>🍿 😊</h1>
                <h2 style="color:#35df91;">POSITIVE REVIEW</h2>
                <p>Your review expresses a positive movie experience!</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="result-card">
                <h1>🎬 😞</h1>
                <h2 style="color:#ff5555;">NEGATIVE REVIEW</h2>
                <p>Your review expresses a negative movie experience.</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        a, b = st.columns(2)

        with a:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

        with b:
            st.metric("Model Accuracy", "90.69%")

        st.markdown("### 📊 Sentiment Probability")

        st.write(f"Negative: {probabilities[0] * 100:.2f}%")
        st.progress(float(probabilities[0]))

        st.write(f"Positive: {probabilities[1] * 100:.2f}%")
        st.progress(float(probabilities[1]))


# ---------------- ABOUT MODEL ----------------

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 Behind the AI</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Training Algorithm", "Logistic Regression")

with c2:
    st.metric("Feature Extraction", "TF-IDF")

with c3:
    st.metric("Test Accuracy", "90.69%")


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    <h3>🎬 CineSense</h3>
    AI-Powered Movie Review Sentiment Analysis
    <br><br>
    Built with Python | Streamlit | Scikit-learn | Machine Learning
    <br>
    © 2026 CineSense
</div>
""", unsafe_allow_html=True)
