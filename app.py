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
st.set_page_config(page_title="Fragrance Verbatim Lab", layout="wide", page_icon="🧪")

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
    "your", "yours", "yourself", "yourselves", "smell", "remind", "think", "is", "have", "may", 
    "should", "also", "bit", "go", "put", "out", "into", "quite", "something", "really", "seem", 
    "evoke", "above", "after", "again", "against", "any", "before", "below", "between", "both", 
    "cannot", "did", "does", "doing", "down", "during", "each", "few", "further", "had", "has", 
    "having", "most", "no", "nor", "off", "once", "only", "other", "over", "own", "same", "some", 
    "such", "than", "then", "through", "under", "up", "was", "were", "therefore", "order", "say", 
    "none", "kind", "kinda", "either", "one", "nothing", "almost", "anything", "everything", "find"
]

# --- Lightweight NLP Setup ---
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

# --- Visualization Logic ---
def get_cloud_mask(shape):
    if shape == "Square":
        return None, (800, 800)
    elif shape == "Rectangle":
        return None, (1000, 500)
    else: # Round
        mask = np.array(Image.new("RGB", (800, 800), (255, 255, 255)))
        d = ImageDraw.Draw(Image.fromarray(mask))
        d.ellipse((10, 10, 790, 790), fill=(0, 0, 0))
        return np.array(d.im), (800, 800)

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series)
    if not combined_text.strip(): return None
    
    mask, dims = get_cloud_mask(shape)
    wc = WordCloud(
        background_color="white", colormap=palette,
        mask=mask, width=dims[0], height=dims[1],
        contour_width=1 if shape == "Round" else 0,
        contour_color='lightgrey'
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_improved_tree(text_series, min_freq, palette_name):
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
        
        # UI Improvements for Readability
        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.kamada_kawai_layout(T, weight='weight') # Physics-based layout
        
        partition = community_louvain.best_partition(T)
        cmap = plt.get_cmap(palette_name)
        
        # Draw edges with varying thickness based on strength
        weights = [G[u][v]['weight'] for u, v in T.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [(w/max_w) * 3 for w in weights]

        nx.draw_networkx_edges(T, pos, alpha=0.2, width=edge_widths, edge_color="grey")
        nx.draw_networkx_nodes(T, pos, node_size=1500, 
                               node_color=[partition[n] for n in T.nodes()], 
                               cmap=cmap, alpha=0.8)
        
        # Add labels with "Halo" effect (better readability)
        nx.draw_networkx_labels(T, pos, font_size=11, font_weight='bold', 
                                font_family='sans-serif')
        
        plt.axis('off')
        return fig
    except: return None

# --- Main App ---
st.title("🧪 Fragrance Verbatim Lab Pro")

if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

with st.sidebar:
    st.header("🎨 Visual Studio")
    shape_opt = st.radio("Word Cloud Shape", ["Rectangle", "Square", "Round"])
    palette_opt = st.selectbox("Color Theme", ["copper", "GnBu", "YlOrBr", "RdPu", "Pastel1"])
    min_freq = st.slider("Word Tree Filter (Min Frequency)", 2, 15, 4)
    
    st.divider()
    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

# Tabs
main_tab, exclusion_tab = st.tabs(["📊 Analysis Lab", "🚫 Word Exclusion List"])

with exclusion_tab:
    st.subheader("Manage Stopwords")
    st.info("These words are completely ignored during analysis. Edit them below.")
    edited_stops = st.text_area("Global Exclusion List (Comma Separated)", 
                                value=", ".join(st.session_state.custom_stop_list), 
                                height=400)
    if st.button("Update Exclusion List"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in edited_stops.split(",") if x.strip()]
        st.success("List updated!")

with main_tab:
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        c1, c2 = st.columns(2)
        p_col = c1.selectbox("Product ID Column", df.columns)
        v_col = c2.selectbox("Verbatim Column", df.columns)

        if st.button("🚀 Process & Generate Visuals"):
            with st.spinner("Cleaning data..."):
                df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
                st.session_state['processed_df'] = df

        if 'processed_df' in st.session_state:
            df = st.session_state['processed_df']
            prod_list = sorted(df[p_col].unique())
            target_prod = st.selectbox("Select Fragrance to Inspect", prod_list)
            
            p_data = df[df[p_col] == target_prod]['cleaned']
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### Cloud View")
                fig_wc = generate_word_cloud(p_data, palette_opt, shape_opt)
                if fig_wc: st.pyplot(fig_wc)
                
            with col_right:
                st.markdown("#### Relationship Tree")
                fig_tree = generate_improved_tree(p_data, min_freq, palette_opt)
                if fig_tree: st.pyplot(fig_tree)

            # Export
            if st.button("📥 Export to PPTX"):
                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                slide.shapes.title.text = f"Scent Map: {target_prod}"
                
                for i, f in enumerate([fig_wc, fig_tree]):
                    if f:
                        img_buf = io.BytesIO()
                        f.savefig(img_buf, format='png', bbox_inches='tight')
                        left = Inches(0.5 + (i*4.75))
                        slide.shapes.add_picture(img_buf, left, Inches(2), width=Inches(4.5))
                
                out = io.BytesIO()
                prs.save(out)
                st.download_button("Download Presentation", out.getvalue(), "Fragrance_Report.pptx")
    else:
        st.warning("Please upload an Excel file to begin.")
