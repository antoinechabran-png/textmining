import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
import re
import numpy as np
from PIL import Image, ImageDraw

# Page Config
st.set_page_config(page_title="Fragrance Lab Pro", layout="wide", page_icon="🧪")

# --- Default Exclusion List ---
DEFAULT_EXCLUSIONS = ["a", "about", "all", "am", "an", "and", "are", "as", "at", "be", "because", "been", "being", "but", "by", "can", "could", "do", "enough", "feel", "for", "from", "have", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "it", "its", "itself", "just", "less", "let", "like", "little", "lot", "make", "me", "more", "my", "myself", "not", "of", "on", "or", "ought", "our", "ours", "ourselves", "product", "real", "she", "should", "so", "that", "the", "their", "theirs", "them", "themselves", "there", "these", "they", "think", "this", "those", "to", "too", "until", "very", "we", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves", "smell", "remind", "is", "may", "also", "bit", "go", "put", "out", "into", "quite", "something", "really", "seem", "evoke", "above", "after", "again", "against", "any", "before", "below", "between", "both", "cannot", "did", "does", "doing", "down", "during", "each", "few", "further", "had", "has", "having", "most", "no", "nor", "off", "once", "only", "other", "over", "own", "same", "some", "such", "than", "then", "through", "under", "up", "was", "were", "therefore", "order", "say", "none", "kind", "kinda", "either", "one", "nothing", "almost", "anything", "everything", "find"]

# --- NLP Engine ---
@st.cache_resource
def setup_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return WordNetLemmatizer()

lemmatizer = setup_nltk()

def clean_text(text, custom_stops):
    if not text or pd.isna(text): return ""
    words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in custom_stops and len(w) > 2]
    return " ".join(cleaned)

# --- Fast Topic Modeling (NMF) ---
def run_fast_topics(text_series, n_topics):
    if len(text_series) < 5: return None, "Insufficient data."
    vectorizer = TfidfVectorizer(max_features=500, stop_words=st.session_state.custom_stop_list)
    tfidf = vectorizer.fit_transform(text_series)
    nmf = NMF(n_components=n_topics, random_state=42).fit(tfidf)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append({"id": topic_idx + 1, "words": top_words})
    return topics, None

# --- UI Functions ---
def generate_word_cloud(text_series, palette, shape, selected_font):
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0)
        mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False)
    wc.generate(" ".join(text_series))
    fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
    return fig

# --- Main App ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data & Style")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    st.divider()
    font_cat = st.selectbox("Font Category", ["Classic", "Modern", "Elegant", "Expressive"])
    selected_font = st.selectbox("Font Style", ["Default"]) # Simplified for speed
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "YlOrBr", "RdPu"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single", "⚔️ Compare", "🌐 FCA", "🔍 Topics", "🚫 Stops"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID", df.columns)
    v_col = st.sidebar.selectbox("Verbatim", df.columns)

    if st.sidebar.button("🚀 Process"):
        df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
        st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())

        with tab4:
            st.subheader("🔍 Scent Theme Discovery")
            n_topics = st.slider("Number of Themes", 2, 6, 3)
            scope = st.radio("Scope", ["All", "Selected Product"], horizontal=True)
            target_data = df['cleaned'] if scope == "All" else df[df[p_col] == st.selectbox("Fragrance", p_list)]
            
            if st.button("Extract Themes"):
                topics, err = run_fast_topics(target_data, n_topics)
                if err: st.error(err)
                else:
                    cols = st.columns(n_topics)
                    for i, topic in enumerate(topics):
                        with cols[i]:
                            st.info(f"**Theme {topic['id']}**")
                            st.write(", ".join(topic['words']))

        # [Other tabs remain same as previous code, simplified for brevity]
