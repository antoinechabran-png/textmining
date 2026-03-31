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
    "kinda", "either", "one", "nothing", "almost", "anything", "everything", "find", "scent", "smell"
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

def generate_word_cloud(text_series, palette, shape, font, disposition):
    combined_text = " ".join(text_series)
    if not combined_text.strip(): return None
    mask, dims = get_cloud_mask(shape)
    
    # 0.0 means only horizontal, 0.5 means 50% vertical/horizontal
    pref_horiz = 1.0 if disposition == "Only Horizontal" else 0.6
    
    wc = WordCloud(
        background_color="white", colormap=palette, mask=mask,
        width=dims[0], height=dims[1], prefer_horizontal=pref_horiz,
        font_path=None, relative_scaling=0.5
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
        # High 'k' and low iterations for clearer bubble separation
        pos = nx.spring_layout(T, k=1.5, iterations=30, seed=42)
        
        partition = community_louvain.best_partition(T)
        cmap = plt.get_cmap(palette_name)
        
        nx.draw_networkx_edges(T, pos, alpha=0.1, edge_color="grey", width=1.5)
        
        # Bubbles (Nodes) - Increased size for readability
        nx.draw_networkx_nodes(T, pos, node_size=3500, 
                               node_color=[partition[n] for n in T.nodes()], 
                               cmap=cmap, alpha=0.8, edgecolors='whitesmoke', linewidths=2)
        
        # Labels - Small font for long words to keep them inside the bubble
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

# Sidebar - Ordered precisely
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

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 Single Product Focus", "⚔️ Comparison Lab", "🚫 Exclusion List"])

with tab3:
    st.subheader("Manage Global Exclusion List")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords", value=current_list, height=450)
    if st.button("Save Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.success("Exclusion list updated!")

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    p_col = st.sidebar.selectbox("Product ID Column", df.columns)
    v_col = st.sidebar.selectbox("Verbatim Column", df.columns)

    if st.sidebar.button("🚀 Run Scent Analysis"):
        with st.spinner("Processing verbatim..."):
            df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
            st.session_state['processed_df'] = df

    if 'processed_df' in st.session_state:
        df = st.session_state['processed_df']
        p_list = sorted(df[p_col].unique())

        with tab1:
            target = st.selectbox("Select Fragrance", p_list, key="single_p")
            p_data = df[df[p_col] == target]['cleaned']
            
            l, r = st.columns(2)
            with l:
                st.markdown("### ☁️ Word Cloud")
                fig_wc = generate_word_cloud(p_data, palette_opt, shape_opt, font_opt, disposition_opt)
                if fig_wc: st.pyplot(fig_wc)
            with r:
                st.markdown("### 🌳 Relationship Tree")
                fig_tree = generate_improved_tree(p_data, min_freq, palette_opt, font_opt)
                if fig_tree: st.pyplot(fig_tree)

            if st.button("📥 Export Analysis to PPTX"):
                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                slide.shapes.title.text = f"Scent Analysis: {target}"
                for i, fig in enumerate([fig_wc, fig_tree]):
                    if fig:
                        img_buf = io.BytesIO()
                        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=200)
                        slide.shapes.add_picture(img_buf, Inches(0.5 + i*4.5), Inches(1.5), width=Inches(4.5))
                out = io.BytesIO()
                prs.save(out)
                st.download_button("Download Report", out.getvalue(), f"{target}_analysis.pptx")

        with tab2:
            st.subheader("Comparison Lab")
            cl1, cl2 = st.columns(2)
            p1 = cl1.selectbox("Product A", p_list, index=0)
            p2 = cl2.selectbox("Product B", p_list, index=min(1, len(p_list)-1))
            
            cl1.pyplot(generate_word_cloud(df[df[p_col]==p1]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt))
            cl2.pyplot(generate_word_cloud(df[df[p_col]==p2]['cleaned'], palette_opt, shape_opt, font_opt, disposition_opt))
            
            cl1.pyplot(generate_improved_tree(df[df[p_col]==p1]['cleaned'], min_freq, palette_opt, font_opt))
            cl2.pyplot(generate_improved_tree(df[df[p_col]==p2]['cleaned'], min_freq, palette_opt, font_opt))
else:
    st.info("Awaiting Excel data to begin lab operations.")
