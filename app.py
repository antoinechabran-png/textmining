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
from collections import Counter
from textblob import TextBlob

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab Pro", layout="wide", page_icon="🧪")

# --- Default Exclusion List ---
DEFAULT_EXCLUSIONS = [
    "a", "about", "all", "am", "an", "and", "are", "as", "at", "be", "because", "been", "being", 
    "but", "by", "can", "could", "do", "enough", "feel", "for", "from", "have", "he", "her", 
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "it", "its", 
    "itself", "just", "less", "let", "like", "little", "lot", "make", "me", "more", "my", 
    "myself", "not", "of", "on", "or", "ought", "our", "ours", "ourselves", "product", "real", 
    "she", "should", "so", "that", "the", "their", "theirs", "them", "themselves", "there", 
    "these", "they", "think", "this", "those", "to", "too", "until", "very", "we", "what", 
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", 
    "your", "yours", "yourself", "yourselves", "smell", "remind", "is", "may", "also", "bit", 
    "go", "put", "out", "into", "quite", "something", "really", "seem", "evoke", "above", 
    "after", "again", "against", "any", "before", "below", "between", "both", "cannot", "did", 
    "does", "doing", "down", "during", "each", "few", "further", "had", "has", "having", "most", 
    "no", "nor", "off", "once", "only", "other", "over", "own", "same", "some", "such", "than", 
    "then", "through", "under", "up", "was", "were", "therefore", "order", "say", "none", "kind", 
    "kinda", "either", "one", "nothing", "almost", "anything", "everything", "find"
]

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

# --- Advanced Analysis Functions ---
def run_keyness(target_series, global_series, top_n=8):
    def get_counts(series):
        words = " ".join(series).split()
        return Counter(words), len(words)
    t_counts, t_total = get_counts(target_series)
    g_counts, g_total = get_counts(global_series)
    if t_total == 0 or g_total == 0: return []
    scores = {w: (c/t_total) / (g_counts.get(w, 1)/g_total) for w, c in t_counts.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

def get_ngrams(text_series, n=2, top_n=8):
    try:
        vec = CountVectorizer(ngram_range=(n, n), stop_words=st.session_state.custom_stop_list)
        mtx = vec.fit_transform(text_series)
        counts = zip(vec.get_feature_names_out(), mtx.toarray().sum(axis=0))
        return sorted(counts, key=lambda x: x[1], reverse=True)[:top_n]
    except: return []

def run_fca(df, p_col, fmin):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products."
    vec = CountVectorizer(min_df=fmin, stop_words=st.session_state.custom_stop_list)
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    total = np.sum(X); row_sums = np.sum(X, axis=1, keepdims=True); col_sums = np.sum(X, axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / total
    Z = (X - expected) / np.sqrt(expected)
    svd = TruncatedSVD(n_components=2); row_coords = svd.fit_transform(Z)
    col_coords = svd.components_.T * (np.std(row_coords) / np.std(svd.components_.T))
    return (row_coords, col_coords, products, words), None

# --- UI Elements ---
def generate_word_cloud(text_series, palette, shape):
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False)
    wc.generate(" ".join(text_series))
    fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); return fig

# --- Main Application ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data & Style")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    st.divider()
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "YlOrBr", "RdPu"])
    st.divider()
    min_freq_tree = st.slider("Tree Depth", 2, 20, 5)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map (FCA)", "🔍 Topic Lab", "🚫 Exclusions"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product Column", df.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df.columns)

    if st.sidebar.button("🚀 Process Data"):
        df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
        st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())

        with tab1:
            target = st.selectbox("Select Fragrance", p_list)
            p_sub = df[df[p_col]==target]['cleaned']
            raw_sub = df[df[p_col]==target][v_col]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("☁️ Word Cloud")
                st.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt))
            with c2:
                # Sentiment Metric
                sent_val = raw_sub.apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
                mood = "Positive" if sent_val > 0.1 else ("Challenging" if sent_val < -0.05 else "Neutral")
                st.metric("Overall Mood", mood, f"{round(sent_val*100, 1)}%")
                st.progress((sent_val + 1) / 2)
                
                st.divider()
                st.subheader("🧬 Scent DNA")
                keys = run_keyness(p_sub, df[df[p_col]!=target]['cleaned'])
                for word, score in keys:
                    sent = TextBlob(word).sentiment.polarity
                    icon = "✨" if sent > 0 else ("⚠️" if sent < 0 else "🏷️")
                    st.write(f"{icon} **{word.upper()}**")
                    st.progress(min(score/10, 1.0))

            st.divider()
            ac1, ac2 = st.columns(2)
            with ac1:
                st.subheader("📜 Bigram Accords")
                for p, c in get_ngrams(p_sub, 2): st.text(f"{c}x | {p}")
            with ac2:
                st.subheader("📜 Trigram Accords")
                for p, c in get_ngrams(p_sub, 3): st.text(f"{c}x | {p}")

        with tab2:
            st.subheader("⚔️ Scent Proximity")
            cl1, cl2 = st.columns(2)
            p1, p2 = cl1.selectbox("Product A", p_list, index=0), cl2.selectbox("Product B", p_list, index=min(1,len(p_list)-1))
            v1, v2 = df[df[p_col]==p1]['cleaned'], df[df[p_col]==p2]['cleaned']
            vec = CountVectorizer(); mtx = vec.fit_transform([" ".join(v1), " ".join(v2)])
            sim = round(cosine_similarity(mtx[0:1], mtx[1:2])[0][0]*100, 1)
            st.metric("Similarity", f"{sim}%")
            cl1.pyplot(generate_word_cloud(v1, palette_opt, shape_opt))
            cl2.pyplot(generate_word_cloud(v2, palette_opt, shape_opt))

        with tab3:
            st.subheader("🌐 FCA Map")
            fmin = st.slider("F-min", 1, 20, 5)
            res, err = run_fca(df, p_col, fmin)
            if not err:
                r_c, c_c, prods, wrds = res
                fig, ax = plt.subplots(figsize=(10,6))
                ax.scatter(r_c[:,0], r_c[:,1], c='blue', alpha=0.5)
                for i, txt in enumerate(prods): ax.annotate(txt, (r_c[i,0], r_c[i,1]))
                ax.scatter(c_c[:,0], c_c[:,1], c='red', marker='x', alpha=0.3)
                st.pyplot(fig)

        with tab4:
            st.subheader("🔍 Topic Lab")
            n_t = st.slider("Themes", 2, 6, 3)
            if st.button("Extract Themes"):
                vec = TfidfVectorizer(max_features=500, stop_words=st.session_state.custom_stop_list)
                mtx = vec.fit_transform(df['cleaned'])
                nmf = NMF(n_components=n_t, random_state=42).fit(mtx)
                fn = vec.get_feature_names_out()
                cols = st.columns(n_t)
                for i, topic in enumerate(nmf.components_):
                    with cols[i]:
                        st.info(f"Theme {i+1}")
                        top_w = [fn[j] for j in topic.argsort()[-5:]]
                        st.write(", ".join(top_w))

with tab5:
    st.subheader("🚫 Exclusions")
    txt = st.text_area("Stopwords", value=", ".join(st.session_state.custom_stop_list), height=300)
    if st.button("Apply"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in txt.split(",") if x.strip()]
        st.rerun()
