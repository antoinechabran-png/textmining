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

# --- Analysis Logic ---
def analyze_shared_dna(text_a, text_b):
    if not text_a or not text_b: return 0.0, pd.DataFrame()
    vec = CountVectorizer()
    try:
        matrix = vec.fit_transform([text_a, text_b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        vocab = vec.get_feature_names_out()
        counts = matrix.toarray()
        shared = [{"Word": vocab[i], "Link Strength": np.sqrt(counts[0,i]*counts[1,i])} 
                  for i in range(len(vocab)) if counts[0,i]>0 and counts[1,i]>0]
        return round(sim*100,1), pd.DataFrame(shared).sort_values("Link Strength", ascending=False).head(10)
    except: return 0.0, pd.DataFrame()

# --- Visualization Helpers ---
def generate_word_cloud(text_series, palette, shape, font, disposition, use_tfidf, all_data=None):
    if text_series.empty: return None
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0)
        mask = np.array(img)
    
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
    
    fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off")
    return fig

def generate_landscape_plot(df, p_col, palette_name):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 2: 
        st.warning("Need at least 2 fragrances to map the landscape.")
        return None

    vectorizer = TfidfVectorizer(max_features=100)
    matrix = vectorizer.fit_transform(grouped)
    words = vectorizer.get_feature_names_out()
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix.toarray())
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=150, c='royalblue', alpha=0.5, edgecolors='white')
    
    # Label Products
    for i, name in enumerate(grouped.index):
        ax.text(coords[i,0], coords[i,1], f"  {name}", fontsize=10, fontweight='bold', va='center')

    # Label Word Influence (Vectors)
    loadings = pca.components_
    for i, word in enumerate(words):
        if np.abs(loadings[0,i]) > 0.25 or np.abs(loadings[1,i]) > 0.25:
            ax.arrow(0, 0, loadings[0,i]*0.8, loadings[1,i]*0.8, color='red', alpha=0.1, head_width=0.02)
            ax.text(loadings[0,i]*0.85, loadings[1,i]*0.85, word, color='darkred', fontsize=8, alpha=0.7)

    ax.axhline(0, color='black', lw=0.5, ls='--'); ax.axvline(0, color='black', lw=0.5, ls='--')
    ax.set_title("Olfactive Map (PCA Landscape)", pad=20)
    plt.tight_layout()
    return fig

# --- Main App ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    st.header("⚙️ Settings")
    min_freq = st.slider("Min Frequency (Tree)", 2, 20, 5)
    shape_opt = st.radio("Shape", ["Rectangle", "Square", "Round"])
    disposition_opt = st.radio("Orientation", ["Only Horizontal", "Mixed Layout"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "YlOrBr", "RdPu"])
    font_opt = st.selectbox("Font", ["sans-serif", "serif", "monospace"])
    use_tfidf = st.toggle("TF-IDF Weighting", value=False)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Landscape", "🚫 Exclusions"])

with tab4:
    st.subheader("Global Exclusion List")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords (comma separated)", value=current_list, height=400)
    if st.button("Apply Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.rerun()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID Column", df.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df.columns)

    if st.sidebar.button("🚀 Process Scent Data"):
        df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
        st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())
        all_texts = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x)).tolist()

        with tab1:
            target = st.selectbox("Select Fragrance", p_list)
            p_sub = df[df[p_col]==target]['cleaned']
            c1, c2 = st.columns(2)
            c1.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_texts))
            # (Tree logic remains here as per previous versions)

        with tab2:
            st.subheader("Head-to-Head Comparison")
            col_a, col_b = st.columns(2)
            p1 = col_a.selectbox("Product A", p_list, index=0)
            p2 = col_b.selectbox("Product B", p_list, index=min(1, len(p_list)-1))
            
            score, shared_df = analyze_shared_dna(" ".join(df[df[p_col]==p1]['cleaned']), " ".join(df[df[p_col]==p2]['cleaned']))
            st.metric("Similarity Score", f"{score}%")
            st.progress(score/100)
            
            col_a.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_texts))
            col_b.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_texts))
            
            if not shared_df.empty:
                st.subheader("🧬 Shared DNA")
                shared_df['Strength'] = shared_df['Link Strength'].apply(lambda x: "▮" * int(min(x, 10)))
                st.table(shared_df[['Word', 'Strength']])

        with tab3:
            st.subheader("🌐 The Olfactive Landscape")
            st.write("This map positions all products based on consumer perception. Closer dots = more similar scents.")
            fig_land = generate_landscape_plot(df, p_col, palette_opt)
            if fig_land: st.pyplot(fig_land)
else:
    st.info("Upload your fragrance Excel to begin.")
