import streamlit as st
import pickle
import re
import string
import pandas as pd
import plotly.express as px
from newspaper import Article
from streamlit_option_menu import option_menu
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

with open("model/fake_news_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# -----------------------------------
# SESSION STATE
# -----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "news_content" not in st.session_state:
    st.session_state.news_content = ""

# -----------------------------------
# CLEAN TEXT
# -----------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)

    return text

# -----------------------------------
# SENTIMENT
# -----------------------------------

def analyze_sentiment(text):

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"

    elif polarity < 0:
        return "Negative"

    else:
        return "Neutral"

# -----------------------------------
# PREDICTION
# -----------------------------------

def predict_news(news):

    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector_input)[0]

    probability = model.predict_proba(vector_input)[0]

    fake_score = probability[0]
    real_score = probability[1]

    return prediction, fake_score, real_score

# -----------------------------------
# URL EXTRACTION
# -----------------------------------

def extract_news_from_url(url):

    article = Article(url)

    article.download()
    article.parse()

    return article.text

# -----------------------------------
# CHECK DUPLICATES
# -----------------------------------

def is_duplicate(news_text):

    cleaned = clean_text(news_text[:100])

    for item in st.session_state.history:

        existing = clean_text(item["News"])

        if cleaned == existing:
            return True

    return False

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    selected = option_menu(
        menu_title="Navigation",
        options=[
            "Home",
            "Detector",
            "Dashboard",
            "About"
        ],
        icons=[
            "house",
            "search",
            "bar-chart",
            "info-circle"
        ],
        default_index=0
    )

# -----------------------------------
# HOME PAGE
# -----------------------------------

if selected == "Home":

    st.title("📰 AI Fake News Detection System")

    st.markdown("""
    ### Detect fake and misleading news using Artificial Intelligence

    ### Features
    ✅ Fake News Detection  
    ✅ URL Article Extraction  
    ✅ Analytics Dashboard  
    ✅ Sentiment Analysis  
    ✅ Trending Keywords  
    ✅ Word Cloud  
    ✅ Duplicate Detection  
    ✅ Delete History  
    """)

    st.image(
        "https://images.unsplash.com/photo-1495020689067-958852a7765e",
        use_container_width=True
    )

# -----------------------------------
# DETECTOR PAGE
# -----------------------------------

elif selected == "Detector":

    st.title("🔍 Fake News Detector")

    input_type = st.radio(
        "Choose Input Type",
        ["Paste News Text", "Paste News URL"]
    )

    # TEXT INPUT

    if input_type == "Paste News Text":

        st.session_state.news_content = st.text_area(
            "Paste News Article Here",
            height=250
        )

    # URL INPUT

    else:

        url = st.text_input("Paste News URL")

        if st.button("Fetch Article"):

            try:

                article_text = extract_news_from_url(url)

                st.session_state.news_content = article_text

                st.success("Article Extracted Successfully")

            except:

                st.error("Unable to fetch article.")

        if st.session_state.news_content != "":

            st.text_area(
                "Extracted Article",
                st.session_state.news_content,
                height=250
            )

    # ANALYZE

    if st.button("Analyze News"):

        news_content = st.session_state.news_content

        if news_content.strip() == "":

            st.warning("Please provide news content.")

        else:

            # DUPLICATE CHECK

            if is_duplicate(news_content):

                st.warning("⚠ This news was already analyzed.")

            else:

                prediction, fake_score, real_score = predict_news(news_content)

                sentiment = analyze_sentiment(news_content)

                if prediction == 0:

                    result = "FAKE"

                    st.error("🚨 FAKE NEWS DETECTED")

                else:

                    result = "REAL"

                    st.success("✅ REAL NEWS")

                # METRICS

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Fake Score",
                    f"{fake_score:.2f}"
                )

                col2.metric(
                    "Real Score",
                    f"{real_score:.2f}"
                )

                col3.metric(
                    "Sentiment",
                    sentiment
                )

                # PIE CHART

                chart_data = pd.DataFrame({
                    "Category": ["Fake", "Real"],
                    "Score": [fake_score, real_score]
                })

                fig = px.pie(
                    chart_data,
                    names="Category",
                    values="Score",
                    color="Category",
                    color_discrete_map={
                        "Fake": "#ff4b4b",
                        "Real": "#00cc96"
                    },
                    title="Prediction Confidence"
                )

                fig.update_traces(
                    textinfo='percent+label',
                    pull=[0.05, 0]
                )

                st.plotly_chart(fig)

                # SAVE HISTORY

                st.session_state.history.append({
                    "News": news_content[:100],
                    "Result": result,
                    "Fake Score": round(fake_score, 2),
                    "Real Score": round(real_score, 2),
                    "Sentiment": sentiment
                })

# -----------------------------------
# DASHBOARD PAGE
# -----------------------------------

elif selected == "Dashboard":

    st.title("📊 Analytics Dashboard")

    history = st.session_state.history

    if len(history) == 0:

        st.info("No predictions available yet.")

    else:

        df = pd.DataFrame(history)

        df.index = range(1, len(df) + 1)

        # METRICS

        total_searches = len(df)

        fake_count = len(df[df["Result"] == "FAKE"])

        real_count = len(df[df["Result"] == "REAL"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Searches", total_searches)

        col2.metric("Fake News", fake_count)

        col3.metric("Real News", real_count)

        st.subheader("📰 Search History")

        # DELETE BUTTONS

        for idx, row in df.iterrows():

            col1, col2 = st.columns([8, 1])

            with col1:

                st.markdown(f"""
                ### {idx}. {row['Result']}

                **News Preview:**  
                {row['News']}

                **Fake Score:** {row['Fake Score']}  
                **Real Score:** {row['Real Score']}  
                **Sentiment:** {row['Sentiment']}
                """)

            with col2:

                if st.button("🗑", key=f"delete_{idx}"):

                    st.session_state.history.pop(idx - 1)

                    st.rerun()

        # BAR CHART

        count_df = pd.DataFrame({
            "Result": ["FAKE", "REAL"],
            "Count": [fake_count, real_count]
        })

        fig = px.bar(
            count_df,
            x="Result",
            y="Count",
            color="Result",
            color_discrete_map={
                "FAKE": "#ff4b4b",
                "REAL": "#00cc96"
            },
            title="Fake vs Real Predictions"
        )

        st.plotly_chart(fig)

        # PIE CHART

        fig2 = px.pie(
            count_df,
            names="Result",
            values="Count",
            color="Result",
            color_discrete_map={
                "FAKE": "#ff4b4b",
                "REAL": "#00cc96"
            },
            title="Prediction Distribution"
        )

        st.plotly_chart(fig2)

        # TRENDING WORDS

        st.subheader("🔥 Trending Keywords")

        all_news = " ".join(df["News"])

        words = all_news.split()

        common_words = Counter(words).most_common(10)

        words_df = pd.DataFrame(
            common_words,
            columns=["Word", "Count"]
        )

        fig3 = px.bar(
            words_df,
            x="Word",
            y="Count",
            title="Top Trending Words"
        )

        st.plotly_chart(fig3)

        # WORD CLOUD

        st.subheader("☁ Word Cloud")

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black'
        ).generate(all_news)

        fig4, ax = plt.subplots()

        ax.imshow(wordcloud, interpolation='bilinear')

        ax.axis("off")

        st.pyplot(fig4)

        # DOWNLOAD REPORT

        csv = df.to_csv(index=True).encode('utf-8')

        st.download_button(
            "📥 Download Prediction Report",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

# -----------------------------------
# ABOUT PAGE
# -----------------------------------

elif selected == "About":

    st.title("ℹ About Project")

    st.markdown("""
    ## AI Fake News Detection System

    This project detects whether a news article is REAL or FAKE using
    Machine Learning and Natural Language Processing.

    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - NLP
    - Plotly
    - TextBlob

    ### Features
    - Fake News Detection
    - URL Article Extraction
    - Analytics Dashboard
    - Duplicate Detection
    - Word Cloud
    - Sentiment Analysis

    ### Accuracy
    Around 95% - 99%
    """)