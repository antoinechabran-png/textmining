import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
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

# --- FCA Core Logic ---
def run_fca(df, p_col, fmin):
    # Group text by product
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3:
        return None, "Need at least 3 fragrances for FCA."

    # Vectorize with Fmin (min_df)
    vec = CountVectorizer(min_df=fmin, stop_words=st.session_state.custom_stop_list)
    X = vec.fit_transform(grouped).toarray()
    words = vec.get_feature_names_out()
    products = grouped.index.tolist()

    if X.shape[1] < 2:
        return None, f"Not enough words meet the threshold Fmin={fmin}. Try lowering the scale."

    # Contingency Table Math (Correspondence Analysis)
    total_sum = np.sum(X)
    row_sums = np.sum(X, axis=1, keepdims=True)
    col_sums = np.sum(X, axis=0, keepdims=True)
    
    expected = (row_sums @ col_sums) / total_sum
    # Chi-square standardized residuals
    Z = (X - expected) / np.sqrt(expected)
    
    # SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=2)
    row_coords = svd.fit_transform(Z)
    col_coords = svd.components_.T

    # Calculate Inertia (Variance Explained)
    var_exp = svd.explained_variance_ratio_ * 100

    return (row_coords, col_coords, products, words, var_exp), None

# --- Visualization Helpers ---
def get_cloud_mask(shape):
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0)
        return np.array(img)
    return None

def generate_word_cloud(text_series, palette, shape, disposition):
    if text_series.empty: return None
    mask = get_cloud_mask(shape)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, 
                   width=800, height=500, stopwords=set(st.session_state.custom_stop_list), 
                   collocations=False, prefer_horizontal=1.0 if disposition=="Only Horizontal" else 0.6)
    wc.generate(" ".join(text_series))
    fig, ax = plt.subplots(); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
    return fig

# --- Main App ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    st.header("🎨 Global Visuals")
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    disposition_opt = st.radio("Orientation", ["Only Horizontal", "Mixed"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "YlOrBr", "RdPu"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map (FCA)", "🚫 Exclusions"])

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

        with tab1:
            target = st.selectbox("Select Fragrance", p_list)
            p_sub = df[df[p_col]==target]['cleaned']
            st.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt, disposition_opt))

        with tab2:
            st.subheader("⚔️ Scent Proximity")
            cl1, cl2 = st.columns(2)
            p1, p2 = cl1.selectbox("Fragrance A", p_list, index=0), cl2.selectbox("Fragrance B", p_list, index=min(1,len(p_list)-1))
            cl1.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, disposition_opt))
            cl2.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, disposition_opt))

        with tab3:
            st.subheader("🌐 Olfactive Correspondence Map")
            # --- FCA Fmin Option ---
            fmin = st.slider("Minimum word frequency threshold ($F_{min}$)", 1, 30, 5, 
                             help="Filters out rare words. Higher values make the map clearer but remove niche descriptors.")
            
            res, error = run_fca(df, p_col, fmin)
            
            if error:
                st.error(error)
            else:
                row_coords, col_coords, products, words, var_exp = res
                
                fig, ax = plt.subplots(figsize=(12, 8))
                # Plot Products
                ax.scatter(row_coords[:, 0], row_coords[:, 1], c='royalblue', s=150, alpha=0.8, label='Products')
                for i, p in enumerate(products):
                    ax.text(row_coords[i, 0], row_coords[i, 1], f"  {p}", fontsize=11, fontweight='bold')
                
                # Plot Words
                ax.scatter(col_coords[:, 0], col_coords[:, 1], c='red', s=40, marker='x', alpha=0.5, label='Words')
                for i, w in enumerate(words):
                    ax.text(col_coords[i, 0], col_coords[i, 1], f" {w}", fontsize=9, color='darkred', alpha=0.7)

                ax.axhline(0, color='grey', lw=1, ls='--')
                ax.axvline(0, color='grey', lw=1, ls='--')
                ax.set_xlabel(f"Factor 1 ({var_exp[0]:.1f}%)")
                ax.set_ylabel(f"Factor 2 ({var_exp[1]:.1f}%)")
                ax.set_title(f"Factorial Correspondence Analysis ($F_{{min}}={fmin}$)")
                st.pyplot(fig)
else:
    st.info("Upload an Excel file to begin.")
