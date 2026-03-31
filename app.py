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

# --- Analysis Functions ---
def run_fca(df, p_col, fmin):
    grouped = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x))
    if len(grouped) < 3: return None, "Need at least 3 products."
    vec = CountVectorizer(min_df=fmin, stop_words=st.session_state.custom_stop_list)
    X = vec.fit_transform(grouped).toarray()
    words = vec.get_feature_names_out()
    products = grouped.index.tolist()
    if X.shape[1] < 2: return None, "Not enough words meet Fmin."
    total = np.sum(X)
    row_sums = np.sum(X, axis=1, keepdims=True); col_sums = np.sum(X, axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / total
    Z = (X - expected) / np.sqrt(expected)
    svd = TruncatedSVD(n_components=2); row_coords = svd.fit_transform(Z)
    col_coords = svd.components_.T * (np.std(row_coords) / np.std(svd.components_.T))
    return (row_coords, col_coords, products, words, svd.explained_variance_ratio_*100), None

def run_fast_topics(text_series, n_topics):
    valid_data = [t for t in text_series if len(str(t).split()) > 2]
    if len(valid_data) < n_topics: return None, "Not enough data for this many themes."
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=st.session_state.custom_stop_list)
    tfidf = vectorizer.fit_transform(valid_data)
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd').fit(tfidf)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for weights in nmf.components_:
        top_indices = weights.argsort()[:-11:-1]
        topics.append([(feature_names[i], weights[i]) for i in top_indices])
    return topics, None

# --- Visualization ---
def generate_word_cloud(text_series, palette, shape, selected_font):
    mask = None
    if shape == "Round":
        img = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(img); draw.ellipse((20,20,780,780), fill=0); mask = np.array(img)
    wc = WordCloud(background_color="white", colormap=palette, mask=mask, width=800, height=500, collocations=False)
    wc.generate(" ".join(text_series))
    fig, ax = plt.subplots(); ax.imshow(wc); ax.axis("off"); return fig

def generate_improved_tree(text_series, min_freq, palette_name):
    valid_text = [t for t in text_series if len(t.split()) > 1]
    if not valid_text: return None
    try:
        vec = CountVectorizer(min_df=min_freq, stop_words=st.session_state.custom_stop_list)
        mtx = vec.fit_transform(valid_text); words = vec.get_feature_names_out()
        adj = (mtx.T * mtx); adj.setdiag(0); G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        T = nx.maximum_spanning_tree(G); fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(T, k=1.6, seed=42); partition = community_louvain.best_partition(T)
        nx.draw_networkx_nodes(T, pos, node_size=2500, node_color=[partition[n] for n in T.nodes()], cmap=plt.get_cmap(palette_name), alpha=0.8)
        nx.draw_networkx_labels(T, pos, font_size=9, font_weight='bold'); nx.draw_networkx_edges(T, pos, alpha=0.1); plt.axis('off'); return fig
    except: return None

# --- UI Setup ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data & Style")
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    st.divider()
    font_cat = st.selectbox("Font Category", ["Classic", "Modern", "Elegant", "Expressive"])
    selected_font = st.selectbox("Font Style", ["Default", "Google Sans", "Roboto", "Inter", "Playfair", "Satoshi"])
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    palette_opt = st.selectbox("Palette", ["copper", "GnBu", "YlOrBr", "RdPu"])
    st.divider()
    min_freq_tree = st.slider("Tree Depth (Min Freq)", 2, 20, 5)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Single Product", "⚔️ Comparison", "🌐 Factorial Map (FCA)", "🔍 Topic Lab", "🚫 Exclusions"])

with tab5:
    st.subheader("Global Exclusion List")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords", value=current_list, height=400)
    if st.button("Apply Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.rerun()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID Column", df.columns)
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
            c1, c2 = st.columns(2)
            with c1: st.pyplot(generate_word_cloud(p_sub, palette_opt, shape_opt, selected_font))
            with c2: st.pyplot(generate_improved_tree(p_sub, min_freq_tree, palette_opt))

        with tab2:
            st.subheader("⚔️ Scent Proximity")
            cl1, cl2 = st.columns(2)
            p1, p2 = cl1.selectbox("Product A", p_list, index=0), cl2.selectbox("Product B", p_list, index=min(1,len(p_list)-1))
            txt_a, txt_b = " ".join(df[df[p_col]==p1]['cleaned']), " ".join(df[df[p_col]==p2]['cleaned'])
            if txt_a and txt_b:
                vec = CountVectorizer(); mtx = vec.fit_transform([txt_a, txt_b])
                sim = round(cosine_similarity(mtx[0:1], mtx[1:2])[0][0]*100, 1)
                st.metric("Similarity Score", f"{sim}%"); st.progress(sim/100)
            cl1.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, selected_font))
            cl2.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, selected_font))

        with tab3:
            st.subheader("🌐 FCA Map")
            fmin_fca = st.slider("Min word frequency ($F_{min}$)", 1, 30, 5)
            res, err = run_fca(df, p_col, fmin_fca)
            if err: st.error(err)
            else:
                row_coords, col_coords, products, words, var_exp = res
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.scatter(row_coords[:, 0], row_coords[:, 1], c='royalblue', s=150, alpha=0.6)
                for i, p in enumerate(products): ax.text(row_coords[i,0], row_coords[i,1], f" {p}", fontweight='bold')
                ax.scatter(col_coords[:, 0], col_coords[:, 1], c='red', marker='x', alpha=0.5)
                for i, w in enumerate(words):
                    if np.linalg.norm(col_coords[i]) > np.mean(np.linalg.norm(col_coords, axis=1)):
                        ax.text(col_coords[i,0], col_coords[i,1], w, color='darkred', fontsize=8)
                st.pyplot(fig)

        with tab4:
            st.subheader("🔍 Topic Lab: Scent Theme Discovery")
            t_col1, t_col2 = st.columns([1, 3])
            with t_col1:
                topic_scope = st.radio("Scope", ["All Products", "Selected Product"])
                if topic_scope == "Selected Product":
                    t_target = st.selectbox("Fragrance", p_list, key="topic_target")
                    data_to_model = df[df[p_col] == t_target]['cleaned']
                else:
                    data_to_model = df['cleaned']
                num_topics = st.slider("Number of Themes", 2, 6, 3)
                run_topic = st.button("🪄 Extract Themes")

            if run_topic:
                topics, err = run_fast_topics(data_to_model, num_topics)
                if err: st.error(err)
                else:
                    t_cols = st.columns(num_topics)
                    for i, topic in enumerate(topics):
                        with t_cols[i]:
                            st.info(f"**Theme {i+1}**")
                            words, weights = zip(*topic)
                            fig, ax = plt.subplots(figsize=(4, 6))
                            ax.barh(words, weights, color=plt.cm.get_cmap(palette_opt)(0.6))
                            ax.invert_yaxis(); ax.set_title(f"Key Descriptors")
                            st.pyplot(fig)
