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
from textblob import TextBlob

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab Pro", layout="wide", page_icon="🧪")

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

# --- Analysis Functions ---
def get_sentiment_words(text_series):
    words = " ".join(text_series).split()
    unique_words = list(set(words))
    scored = [(w, TextBlob(w).sentiment.polarity) for w in unique_words]
    pos = sorted([x for x in scored if x[1] > 0], key=lambda x: x[1], reverse=True)[:10]
    neg = sorted([x for x in scored if x[1] < 0], key=lambda x: x[1])[:10]
    return pos, neg

def run_fca(df, p_col, fmin, use_tfidf):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need 3+ products."
    
    VecClass = TfidfVectorizer if use_tfidf else CountVectorizer
    vec = VecClass(min_df=fmin, stop_words=st.session_state.custom_stop_list)
    X = vec.fit_transform(grouped).toarray()
    words, products = vec.get_feature_names_out(), grouped.index.tolist()
    
    svd = TruncatedSVD(n_components=2)
    row_coords = svd.fit_transform(X)
    col_coords = svd.components_.T * (np.std(row_coords) / np.std(svd.components_.T))
    return (row_coords, col_coords, products, words), None

def generate_word_cloud(text_series, palette, shape):
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False)
    wc.generate(" ".join(text_series))
    fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); return fig

def generate_word_tree(text_series, min_freq, palette):
    valid = [t for t in text_series if len(t.split()) > 1]
    if not valid: return None
    try:
        vec = CountVectorizer(min_df=min_freq, stop_words=st.session_state.custom_stop_list)
        mtx = vec.fit_transform(valid); words = vec.get_feature_names_out()
        adj = (mtx.T * mtx); adj.setdiag(0); G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        T = nx.maximum_spanning_tree(G)
        fig, ax = plt.subplots(figsize=(8,6))
        pos = nx.spring_layout(T, k=1.5, seed=42); part = community_louvain.best_partition(T)
        nx.draw_networkx_nodes(T, pos, node_size=2000, node_color=list(part.values()), cmap=palette, alpha=0.8)
        nx.draw_networkx_labels(T, pos, font_size=8, font_weight='bold'); nx.draw_networkx_edges(T, pos, alpha=0.2)
        plt.axis('off'); return fig
    except: return None

# --- UI Setup ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

with st.sidebar:
    st.header("⚙️ Global Settings")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    use_tfidf = st.toggle("Use TF-IDF Weighting", value=True)
    fmin_global = st.slider("Min Word Frequency (Global)", 1, 50, 5)
    st.divider()
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Round"])
    palette_opt = st.selectbox("Color Palette", ["copper", "GnBu", "RdPu", "viridis"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map", "🔍 Topic Lab", "🚫 Exclusions"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID", df.columns)
    v_col = st.sidebar.selectbox("Verbatim", df.columns)

    if st.sidebar.button("🚀 Run Analysis"):
        df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
        st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())

        with tab1:
            target = st.selectbox("Fragrance Focus", p_list)
            p_sub = df[df[p_col]==target]['cleaned']
            
            # 1. Overall Mood Score
            sent_val = df[df[p_col]==target][v_col].apply(lambda x: TextBlob(str(x)).sentiment.polarity).mean()
            st.metric("Overall Brand Mood", f"{'Positive' if sent_val > 0 else 'Negative'}", f"{round(sent_val*100, 1)}%")
            st.progress((sent_val + 1) / 2)
            st.divider()

            # 2. Visuals Row
            c1, c2 = st.columns(2)
            with c1: 
                st.write("**Word Cloud**")
                st.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt))
            with c2: 
                st.write("**Word Tree (Scent Accords)**")
                tree_fig = generate_word_tree(p_sub, fmin_global, palette_opt)
                if tree_fig: st.pyplot(tree_fig)

            # 3. Sentiment Word Lists
            st.divider()
            pos_words, neg_words = get_sentiment_words(p_sub)
            l_col, r_col = st.columns(2)
            with l_col:
                st.success("✨ **Top 10 Positive Descriptors**")
                for w, s in pos_words: st.write(f"- {w}")
            with r_col:
                st.error("⚠️ **Top 10 Negative/Challenging Descriptors**")
                for w, s in neg_words: st.write(f"- {w}")

            # 4. N-Grams at the very end
            st.divider()
            st.write("📜 **Detailed Phrase extractions (N-Grams)**")
            vec2 = CountVectorizer(ngram_range=(2,3), stop_words=st.session_state.custom_stop_list)
            try:
                mtx_n = vec2.fit_transform(p_sub)
                ng_counts = zip(vec2.get_feature_names_out(), mtx_n.toarray().sum(axis=0))
                for p, c in sorted(ng_counts, key=lambda x: x[1], reverse=True)[:10]:
                    st.caption(f"{c}x | {p}")
            except: st.write("Not enough phrases found.")

        with tab3:
            st.subheader("🌐 Factorial Correspondence Analysis")
            res, err = run_fca(df, p_col, fmin_global, use_tfidf)
            if not err:
                r_c, c_c, prods, wrds = res
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.scatter(r_c[:,0], r_c[:,1], c='blue', s=100, label="Products")
                for i, txt in enumerate(prods): ax.text(r_c[i,0], r_c[i,1], txt, fontweight='bold', fontsize=10)
                
                # Filter words to show only those further from center (significant)
                ax.scatter(c_c[:,0], c_c[:,1], c='red', marker='x', alpha=0.4, label="Words")
                for i, txt in enumerate(wrds):
                    if np.linalg.norm(c_c[i]) > np.mean(np.linalg.norm(c_c, axis=1)):
                        ax.text(c_c[i,0], c_c[i,1], txt, color='red', alpha=0.7, fontsize=8)
                
                ax.axhline(0, color='black', lw=1, ls='--'); ax.axvline(0, color='black', lw=1, ls='--')
                st.pyplot(fig)

        with tab4:
            st.subheader("🔍 Topic Lab (Scent Story Discovery)")
            num_t = st.slider("Number of Themes", 2, 8, 4)
            if st.button("Generate Themes"):
                # Use TF-IDF for better topic differentiation
                vec_t = TfidfVectorizer(max_features=1000, stop_words=st.session_state.custom_stop_list)
                mtx_t = vec_t.fit_transform(df['cleaned'])
                nmf = NMF(n_components=num_t, random_state=42).fit(mtx_t)
                feature_names = vec_t.get_feature_names_out()
                
                for i, topic in enumerate(nmf.components_):
                    with st.expander(f"Theme {i+1} : {', '.join([feature_names[j] for j in topic.argsort()[-3:]])}"):
                        top_words = [feature_names[j] for j in topic.argsort()[-10:]]
                        st.write("Keywords: " + ", ".join(top_words))
                        # Find top representative fragrance for this theme
                        relevance = mtx_t.dot(topic)
                        top_prod_idx = relevance.argmax()
                        st.caption(f"Representative Product: {df.iloc[top_prod_idx][p_col]}")

with tab5:
    st.subheader("🚫 Exclusions")
    txt = st.text_area("Stopwords", value=", ".join(st.session_state.custom_stop_list), height=300)
    if st.button("Apply"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in txt.split(",") if x.strip()]
        st.rerun()
