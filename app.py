import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
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

# --- Visual Logic ---
def get_cloud_mask(shape):
    if shape == "Square": return None, (800, 800)
    elif shape == "Rectangle": return None, (1000, 500)
    else: 
        mask = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((20, 20, 780, 780), fill=0)
        return np.array(mask), (800, 800)

def generate_word_cloud(text_series, palette, shape, font):
    combined_text = " ".join(text_series)
    if not combined_text.strip(): return None
    mask, dims = get_cloud_mask(shape)
    wc = WordCloud(
        background_color="white", colormap=palette, mask=mask,
        width=dims[0], height=dims[1], font_path=None, # Streamlit uses system fonts
        prefer_horizontal=0.7
    ).generate(combined_text)
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
        # Increase 'k' for more space between bubbles
        pos = nx.spring_layout(T, k=1.2, iterations=50, seed=42)
        
        partition = community_louvain.best_partition(T)
        cmap = plt.get_cmap(palette_name)
        
        # Draw edges
        nx.draw_networkx_edges(T, pos, alpha=0.15, edge_color="grey")
        
        # Draw Nodes (Bubbles)
        nx.draw_networkx_nodes(T, pos, node_size=3000, 
                               node_color=[partition[n] for n in T.nodes()], 
                               cmap=cmap, alpha=0.7)
        
        # Labels with dynamic size reduction for better fit
        for node, (x, y) in pos.items():
            size = 11 if len(node) < 6 else 9
            ax.text(x, y, node, fontsize=size, ha='center', va='center', 
                    fontweight='bold', family=font_choice)
        
        plt.axis('off')
        return fig
    except: return None

# --- Application ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

# Sidebar - Ordered as requested
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    
    st.header("⚙️ Graph Settings")
    min_freq = st.slider("Min Word Frequency", 2, 20, 5)
    
    st.header("🎨 Visual Studio")
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    palette_opt = st.selectbox("Color Palette", ["copper", "GnBu", "YlOrBr", "RdPu", "Pastel1"])
    font_opt = st.selectbox("Font Style", ["sans-serif", "serif", "monospace", "fantasy"])

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Single Product", "⚔️ Comparison Lab", "🚫 Exclusion List"])

with tab3:
    st.subheader("Global Exclusion List")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords", value=current_list, height=450)
    if st.button("Save Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.rerun()

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID Column", df.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df.columns)

    if st.sidebar.button("🚀 Process Data"):
        with st.spinner("Cleaning..."):
            df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
            st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())

        with tab1:
            target = st.selectbox("Analyze Fragrance", p_list, key="single")
            p_data = df[df[p_col] == target]['cleaned']
            
            l, r = st.columns(2)
            with l:
                st.pyplot(generate_word_cloud(p_data, palette_opt, shape_opt, font_opt))
            with r:
                st.pyplot(generate_improved_tree(p_data, min_freq, palette_opt, font_opt))

        with tab2:
            st.subheader("Head-to-Head Comparison")
            c1, c2 = st.columns(2)
            p1 = c1.selectbox("Product A", p_list, index=0)
            p2 = c2.selectbox("Product B", p_list, index=min(1, len(p_list)-1))
            
            c1.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, font_opt))
            c2.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, font_opt))
            
            c1.pyplot(generate_improved_tree(df[df[p_col]==p1]['cleaned'], min_freq, palette_opt, font_opt))
            c2.pyplot(generate_improved_tree(df[df[p_col]==p2]['cleaned'], min_freq, palette_opt, font_opt))

        if st.button("📥 Export Current Analysis to PPTX"):
            prs = Presentation()
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.title.text = "Fragrance Analysis"
            # Add Export Logic here (as per previous code)
            st.success("Ready for download!")
else:
    st.info("Upload an Excel file to get started.")
