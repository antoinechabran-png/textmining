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
    # Extract only alphabetic words longer than 2 chars
    words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in custom_stops])

# --- Visual Logic ---
def get_cloud_mask(shape):
    if shape == "Square":
        return None, (800, 800)
    elif shape == "Rectangle":
        return None, (1000, 500)
    else: # Round
        # Create a circular mask using Pillow
        mask = Image.new("L", (800, 800), 255)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((20, 20, 780, 780), fill=0)
        return np.array(mask), (800, 800)

def generate_word_cloud(text_series, palette, shape):
    combined_text = " ".join(text_series)
    if not combined_text.strip(): return None
    
    mask, dims = get_cloud_mask(shape)
    
    wc = WordCloud(
        background_color="white",
        colormap=palette,
        mask=mask,
        width=dims[0],
        height=dims[1],
        contour_width=2 if shape == "Round" else 0,
        contour_color='whitesmoke',
        prefer_horizontal=0.7
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_improved_tree(text_series, min_freq, palette_name):
    # Only use descriptions with more than 1 word for relationship building
    valid_text = [t for t in text_series if len(t.split()) > 1]
    if not valid_text: return None

    try:
        vec = CountVectorizer(min_df=min_freq)
        mtx = vec.fit_transform(valid_text)
        words = vec.get_feature_names_out()
        
        # Build Co-occurrence Graph
        adj = (mtx.T * mtx)
        adj.setdiag(0)
        
        G = nx.from_scipy_sparse_array(adj)
        G = nx.relabel_nodes(G, {i: w for i, w in enumerate(words)})
        
        # Use MST to prevent hairballs
        T = nx.maximum_spanning_tree(G)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        # Layout: Kamada-Kawai provides much clearer spacing for small trees
        pos = nx.kamada_kawai_layout(T)
        
        partition = community_louvain.best_partition(T)
        cmap = plt.get_cmap(palette_name)
        
        # Edge width logic for better visibility
        weights = [G[u][v]['weight'] for u, v in T.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [(w/max_w) * 4 for w in weights]

        nx.draw_networkx_edges(T, pos, alpha=0.3, width=edge_widths, edge_color="lightgrey")
        nx.draw_networkx_nodes(T, pos, node_size=1800, 
                               node_color=[partition[n] for n in T.nodes()], 
                               cmap=cmap, alpha=0.9, edgecolors='white', linewidths=1)
        
        # Label formatting
        nx.draw_networkx_labels(T, pos, font_size=12, font_weight='bold')
        
        plt.axis('off')
        return fig
    except Exception:
        return None

# --- Application UI ---
if 'custom_stop_list' not in st.session_state:
    st.session_state.custom_stop_list = DEFAULT_EXCLUSIONS.copy()

st.title("🧪 Fragrance Verbatim Lab Pro")

with st.sidebar:
    st.header("🎨 Canvas Settings")
    shape_opt = st.radio("Cloud Shape", ["Rectangle", "Square", "Round"])
    palette_opt = st.selectbox("Color Palette", ["copper", "GnBu", "YlOrBr", "RdPu", "Pastel1"])
    min_freq = st.slider("Tree Detail (Min Freq)", 2, 20, 5)
    
    st.divider()
    uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

# Layout Tabs
main_tab, exclusion_tab = st.tabs(["📊 Scent Analysis", "🚫 Exclusion List"])

with exclusion_tab:
    st.subheader("Global Exclusion List")
    st.write("Edit the list below to filter out generic terms.")
    current_list = ", ".join(st.session_state.custom_stop_list)
    updated_input = st.text_area("Stopwords", value=current_list, height=450)
    if st.button("Save Changes"):
        st.session_state.custom_stop_list = [x.strip().lower() for x in updated_input.split(",") if x.strip()]
        st.rerun()

with main_tab:
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        cols = df.columns.tolist()
        
        c1, c2 = st.columns(2)
        p_col = c1.selectbox("Product ID Column", cols)
        v_col = c2.selectbox("Verbatim Column", cols)

        if st.button("🚀 Run Analysis"):
            with st.spinner("Processing feedback..."):
                df['cleaned'] = df[v_col].apply(lambda x: clean_text(x, st.session_state.custom_stop_list))
                st.session_state['processed_df'] = df

        if 'processed_df' in st.session_state:
            df = st.session_state['processed_df']
            p_list = sorted(df[p_col].unique())
            target = st.selectbox("Fragrance to Analyze", p_list)
            
            p_subset = df[df[p_col] == target]['cleaned']
            
            left, right = st.columns(2)
            with left:
                st.markdown("### ☁️ Scent Word Cloud")
                fig_wc = generate_word_cloud(p_subset, palette_opt, shape_opt)
                if fig_wc: st.pyplot(fig_wc)
                
            with right:
                st.markdown("### 🌳 Scent Relationship Tree")
                fig_tree = generate_improved_tree(p_subset, min_freq, palette_opt)
                if fig_tree: st.pyplot(fig_tree)

            # PPTX Export
            if st.button("📥 Download PPTX Report"):
                prs = Presentation()
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                slide.shapes.title.text = f"Consumer Scent Map: {target}"
                
                # Add images to slide
                for i, fig in enumerate([fig_wc, fig_tree]):
                    if fig:
                        img_io = io.BytesIO()
                        fig.savefig(img_io, format='png', bbox_inches='tight', dpi=300)
                        left_pos = Inches(0.5 + (i * 4.75))
                        slide.shapes.add_picture(img_io, left_pos, Inches(1.5), width=Inches(4.5))
                
                ppt_io = io.BytesIO()
                prs.save(ppt_io)
                st.download_button("Click to Download", ppt_io.getvalue(), f"{target}_Report.pptx")
    else:
        st.info("Upload your fragrance data to begin visualization.")
