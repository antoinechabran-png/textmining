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
from sklearn.decomposition import PCA
from pptx import Presentation
from pptx.util import Inches
import io
import re
import numpy as np
from PIL import Image, ImageDraw

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
    cleaned = []
    for w in words:
        if w in custom_stops: continue
        lemma = lemmatizer.lemmatize(w)
        if lemma not in custom_stops and len(lemma) > 2:
            cleaned.append(lemma)
    return " ".join(cleaned)

# --- Visual Logic ---
def get_cloud_mask(shape):
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0)
        return np.array(img)
    return None

def generate_word_cloud(text_series, palette, shape, font, disposition, use_tfidf, all_data=None):
    if text_series.empty: return None
    mask = get_cloud_mask(shape)
    
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, 
                   width=800, height=500 if shape=="Rectangle" else 800,
                   stopwords=set(st.session_state.custom_stop_list), collocations=False,
                   prefer_horizontal=1.0 if disposition=="Only Horizontal" else 0.6)
    
    if use_tfidf and all_data:
        tfidf = TfidfVectorizer(stop_words=st.session_state.custom_stop_list)
        tfidf.fit(all_data)
        res = tfidf.transform([" ".join(text_series)])
        wc.generate_from_frequencies({tfidf.get_feature_names_out()[i]: res[0,i] for i in res.indices})
    else:
        wc.generate(" ".join(text_series))
    
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

def generate_improved_tree(text_series, min_freq, palette_name, font_choice):
    valid_text = [t for t in text_series if len(t.split()) > 1]
    if not valid_text: return None
    try:
        vec = CountVectorizer(min_df=min_freq, stop_words=st.session_state.custom_stop_list)
        mtx = vec.fit_transform(valid_text)
        words = vec.get_feature_names_out()
        adj = (mtx.T * mtx); adj.setdiag(0)
        G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        T = nx.maximum_spanning_tree(G)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        pos = nx.spring_layout(T, k=1.6, iterations=45, seed=42)
        partition = community_louvain.best_partition(T)
        cmap = plt.get_cmap(palette_name)
        
        nx.draw_networkx_edges(T, pos, alpha=0.1, edge_color="grey")
        nx.draw_networkx_nodes(T, pos, node_size=3800, 
                               node_color=[partition[n] for n in T.nodes()], 
                               cmap=cmap, alpha=0.8, edgecolors='whitesmoke', linewidths=2)
        
        for node, (x, y) in pos.items():
            f_size = 11 if len(node) < 7 else 9
            ax.text(x, y, node, fontsize=f_size, ha='center', va='center', fontweight='bold', 
                    family=font_choice, bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'))
        plt.axis('off')
        return fig
    except: return None

def generate_landscape_plot(df, p_col):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 2: return None
    vectorizer = TfidfVectorizer(max_features=100, stop_words=st.session_state.custom_stop_list)
    matrix = vectorizer.fit_transform(grouped)
    pca = PCA(n_components=2); coords = pca.fit_transform(matrix.toarray())
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=150, c='royalblue', alpha=0.5)
    for i, name in enumerate(grouped.index):
        ax.text(coords[i,0], coords[i,1], f" {name}", fontsize=9, fontweight='bold')
    
    loadings = pca.components_; words = vectorizer.get_feature_names_out()
    for i, word in enumerate(words):
        if np.abs(loadings[0,i]) > 0.25 or np.abs(loadings[1,i]) > 0.25:
            ax.arrow(0, 0, loadings[0,i]*0.7, loadings[1,i]*0.7, color='red', alpha=0.1)
            ax.text(loadings[0,i]*0.75, loadings[1,i]*0.75, word, color='darkred', fontsize=8)
    ax.axhline(0, color='black', lw=0.5, ls='--'); ax.axvline(0, color='black', lw=0.5, ls='--')
    return fig

# --- App UI ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    st.header("⚙️ Detail")
    min_freq = st.slider("Min Frequency (Tree)", 2, 20, 5)
    st.header("🎨 Visuals")
    shape_opt = st.radio("Shape", ["Rectangle", "Square", "Round"])
    disposition_opt = st.radio("Orientation", ["Only Horizontal", "Mixed"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "YlOrBr", "RdPu", "Pastel1"])
    font_opt = st.selectbox("Font", ["sans-serif", "serif", "monospace"])
    use_tfidf = st.toggle("TF-IDF Weighting", value=False)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Landscape", "🚫 Exclusions"])

with tab4:
    st.subheader("Global Exclusion List")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords", value=current_list, height=400)
    if st.button("Apply Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.rerun()

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
        all_txt = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x)).tolist()

        with tab1:
            target = st.selectbox("Select Fragrance", p_list)
            p_sub = df[df[p_col]==target]['cleaned']
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### ☁️ Word Cloud")
                st.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_txt))
            with c2:
                st.markdown("### 🌳 Relationship Tree")
                st.pyplot(generate_improved_tree(p_sub, min_freq, palette_opt, font_opt))

        with tab2:
            st.subheader("⚔️ Comparison & Proximity")
            cl1, cl2 = st.columns(2)
            p1, p2 = cl1.selectbox("Fragrance A", p_list, index=0), cl2.selectbox("Fragrance B", p_list, index=min(1,len(p_list)-1))
            txt_a, txt_b = " ".join(df[df[p_col]==p1]['cleaned']), " ".join(df[df[p_col]==p2]['cleaned'])
            vec = CountVectorizer(); mtx = vec.fit_transform([txt_a, txt_b])
            sim = round(cosine_similarity(mtx[0:1], mtx[1:2])[0][0]*100, 1)
            st.metric("Similarity Score", f"{sim}%"); st.progress(sim/100)
            cl1.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_txt))
            cl2.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_txt))

        with tab3:
            st.subheader("🌐 Olfactive Landscape")
            fig_land = generate_landscape_plot(df, p_col)
            if fig_land: st.pyplot(fig_land)
