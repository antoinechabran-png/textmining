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
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in custom_stops])

# --- Logic for Proximity & Shared DNA ---
def analyze_shared_dna(text_a, text_b):
    if not text_a or not text_b: return 0.0, pd.DataFrame()
    
    vec = CountVectorizer()
    matrix = vec.fit_transform([text_a, text_b])
    vocab = vec.get_feature_names_out()
    
    # 1. Similarity Score
    sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    
    # 2. Shared Words Analysis
    counts_a = matrix.toarray()[0]
    counts_b = matrix.toarray()[1]
    
    shared_data = []
    for i, word in enumerate(vocab):
        if counts_a[i] > 0 and counts_b[i] > 0:
            # Strength is the geometric mean of frequencies
            strength = np.sqrt(counts_a[i] * counts_b[i])
            shared_data.append({"Word": word, "Link Strength": strength})
    
    df_shared = pd.DataFrame(shared_data).sort_values(by="Link Strength", ascending=False).head(10)
    return round(sim * 100, 1), df_shared

# --- Visual Logic ---
def get_cloud_mask(shape):
    if shape == "Square": return None, (800, 800)
    elif shape == "Rectangle": return None, (1000, 500)
    else: 
        mask = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((20, 20, 780, 780), fill=0)
        return np.array(mask), (800, 800)

def generate_word_cloud(text_series, palette, shape, font, disposition, use_tfidf, all_data=None):
    if text_series.empty or not " ".join(text_series).strip(): return None
    mask, dims = get_cloud_mask(shape)
    pref_horiz = 1.0 if disposition == "Only Horizontal" else 0.6
    
    if use_tfidf and all_data is not None:
        tfidf = TfidfVectorizer()
        tfidf.fit(all_data)
        feature_names = tfidf.get_feature_names_out()
        prod_text = " ".join(text_series)
        response = tfidf.transform([prod_text])
        frequencies = {feature_names[i]: response[0, i] for i in response.indices}
    else:
        combined_text = " ".join(text_series)
        frequencies = WordCloud().process_text(combined_text)

    wc = WordCloud(
        background_color="white", colormap=palette, mask=mask,
        width=dims[0], height=dims[1], prefer_horizontal=pref_horiz,
        relative_scaling=0.5
    ).generate_from_frequencies(frequencies)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_improved_tree(text_series, min_freq, palette_name, font_choice):
    valid_text = [t for t in text_series if len(t.split()) > 1]
    if not valid_text: return None
    try:
        vec = CountVectorizer(min_df=min_freq)
        mtx = vec.fit_transform(valid_text)
        words = vec.get_feature_names_out()
        adj = (mtx.T * mtx)
        adj.setdiag(0)
        G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        T = nx.maximum_spanning_tree(G)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(T, k=1.5, iterations=30, seed=42)
        partition = community_louvain.best_partition(T)
        cmap = plt.get_cmap(palette_name)
        
        nx.draw_networkx_edges(T, pos, alpha=0.1, edge_color="grey")
        nx.draw_networkx_nodes(T, pos, node_size=3500, 
                               node_color=[partition[n] for n in T.nodes()], 
                               cmap=cmap, alpha=0.8, edgecolors='whitesmoke', linewidths=2)
        
        for node, (x, y) in pos.items():
            font_size = 10 if len(node) < 8 else 8
            ax.text(x, y, node, fontsize=font_size, ha='center', va='center', 
                    fontweight='bold', family=font_choice,
                    bbox=dict(facecolor='white', alpha=0.1, edgecolor='none', pad=1))
        plt.axis('off')
        return fig
    except: return None

# --- Application ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    
    st.header("⚙️ Graph Detail")
    min_freq = st.slider("Min Word Frequency (Tree)", 2, 20, 5)
    
    st.header("🎨 Visual Studio")
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    disposition_opt = st.radio("Word Orientation", ["Only Horizontal", "Mixed Layout"])
    palette_opt = st.selectbox("Color Palette", ["copper", "GnBu", "YlOrBr", "RdPu", "Pastel1"])
    font_opt = st.selectbox("Font Style", ["sans-serif", "serif", "monospace"])
    
    st.header("🔬 Advanced NLP")
    use_tfidf = st.toggle("Activate TF-IDF Weighting", value=False)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Single Product", "⚔️ Comparison Lab", "🚫 Exclusion List"])

with tab3:
    st.subheader("Manage Global Exclusion List")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords", value=current_list, height=450)
    if st.button("Save Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.rerun()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID Column", df.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df.columns)

    if st.sidebar.button("🚀 Run Scent Analysis"):
        with st.spinner("Processing..."):
            df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
            st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())
        all_cleaned_texts = df.groupby(p_col)['cleaned'].apply(lambda x: " ".join(x)).tolist()

        with tab1:
            target = st.selectbox("Select Fragrance", p_list, key="single_p")
            p_data = df[df[p_col] == target]['cleaned']
            l, r = st.columns(2)
            with l:
                st.pyplot(generate_word_cloud(p_data, palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_cleaned_texts))
            with r:
                st.pyplot(generate_improved_tree(p_data, min_freq, palette_opt, font_opt))

        with tab2:
            st.subheader("⚔️ Scent Comparison & Proximity Score")
            p_comp_1, p_comp_2 = st.columns(2)
            p1 = p_comp_1.selectbox("Fragrance A", p_list, index=0)
            p2 = p_comp_2.selectbox("Fragrance B", p_list, index=min(1, len(p_list)-1))
            
            # Data prep
            text_a = " ".join(df[df[p_col]==p1]['cleaned'])
            text_b = " ".join(df[df[p_col]==p2]['cleaned'])
            
            # Similarity Logic
            score, shared_df = analyze_shared_dna(text_a, text_b)
            
            # Similarity UI (BEFORE Clouds)
            st.write(f"### Olfactory Similarity: **{score}%**")
            st.progress(score / 100)
            
            st.divider()
            
            # Word Clouds
            col_cloud_a, col_cloud_b = st.columns(2)
            with col_cloud_a:
                st.write(f"**{p1}** Profile")
                st.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_cleaned_texts))
            with col_cloud_b:
                st.write(f"**{p2}** Profile")
                st.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt, use_tfidf, all_cleaned_texts))
            
            st.divider()
            
            # Shared DNA (AFTER Clouds)
            st.subheader("🧬 Shared Olfactive DNA")
            st.write("Top 10 shared descriptors contributing to the similarity score.")
            
            if not shared_df.empty:
                # Format weight for display
                shared_df['Link Strength'] = shared_df['Link Strength'].apply(lambda x: "▮" * int(min(x, 10)))
                st.table(shared_df)
            else:
                st.warning("No shared descriptors found between these two products.")

else:
    st.info("Upload your Excel file to begin.")
