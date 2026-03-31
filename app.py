import streamlit as st
import pandas as pd
import spacy
import networkx as nx
from community import community_louvain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from pptx import Presentation
from pptx.util import Inches
import io
import re

# Page Config
st.set_page_config(page_title="Fragrance Verbatim Lab", layout="wide", page_icon="🧪")

# --- Optimized NLP Setup ---
@st.cache_resource
def load_nlp():
    # Loading only the tokenizer and lemmatizer to save 60-70% load time
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "entity_linker", "textcat"])
        return nlp
    except Exception:
        # Fallback if model link is different in environment
        return spacy.blank("en")

nlp = load_nlp()

def clean_text(text, custom_stopwords):
    if not text or pd.isna(text):
        return ""
    # Process text through the light pipe
    doc = nlp(str(text).lower())
    # Filter: Is alpha, not a standard stopword, not in custom list, length > 2
    tokens = [token.lemma_ for token in doc if token.is_alpha 
              and not token.is_stop 
              and token.lemma_ not in custom_stopwords
              and len(token.lemma_) > 2]
    return " ".join(tokens)

# --- Visualization Engines ---
def generate_word_cloud(text_series, palette, min_freq):
    combined_text = " ".join(text_series)
    if not combined_text.strip():
        return None
    
    wc = WordCloud(
        background_color="white",
        colormap=palette,
        min_font_size=10,
        max_words=100,
        width=800, height=500
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_word_tree(text_series, min_freq, palette_name):
    # Filter empty or tiny strings
    valid_text = [t for t in text_series if len(t.split()) > 1]
    if not valid_text:
        return None

    try:
        # Co-occurrence Matrix
        vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=min_freq)
        sparse_matrix = vectorizer.fit_transform(valid_text)
        words = vectorizer.get_feature_names_out()
        
        # Adjacency matrix: (Words x Words)
        adj_matrix = (sparse_matrix.T * sparse_matrix)
        adj_matrix.setdiag(0)
        
        G = nx.from_scipy_sparse_array(adj_matrix)
        mapping = {i: word for i, word in enumerate(words)}
        G = nx.relabel_nodes(G, mapping)
        
        if len(G.nodes) < 2:
            return None

        # Community Detection (Louvain)
        partition = community_louvain.best_partition(G)
        
        # Maximum Spanning Tree (creates the 'Branching' effect)
        T = nx.maximum_spanning_tree(G)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.kamada_kawai_layout(T) # Better for "Tree" structures than Spring
        
        # Map colors from palette
        cmap = plt.get_cmap(palette_name)
        node_colors = [partition[node] for node in T.nodes()]
        
        nx.draw_networkx_nodes(T, pos, node_size=1000, node_color=node_colors, cmap=cmap, alpha=0.9)
        nx.draw_networkx_labels(T, pos, font_size=9, font_weight="bold")
        nx.draw_networkx_edges(T, pos, alpha=0.2, edge_color="gray")
        
        plt.axis('off')
        return fig
    except Exception as e:
        st.error(f"Graph Error: {e}")
        return None

# --- UI Sidebar ---
st.sidebar.title("🚀 Fragrance Lab")
uploaded_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

st.sidebar.subheader("⚙️ NLP Filters")
custom_stopwords_input = st.sidebar.text_area(
    "Custom Stopwords", 
    "smell, product, think, fragrance, perfume, like, feel, really, bit"
)
custom_stop_list = [x.strip().lower() for x in custom_stopwords_input.split(",")]
min_freq = st.sidebar.slider("Minimum Word Frequency", 2, 10, 3)

st.sidebar.subheader("🎨 UI Design")
palettes = {
    "Pastel": "Pastel1", 
    "Woody": "copper", 
    "Fresh": "GnBu", 
    "Citrus": "YlOrBr", 
    "Floral": "RdPu",
    "Deep Sea": "coolwarm"
}
selected_palette = st.sidebar.selectbox("Color Palette", list(palettes.keys()))

# --- Main App Logic ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    cols = df.columns.tolist()
    
    st.success(f"Loaded {len(df)} responses.")
    
    c1, c2 = st.columns(2)
    with c1:
        prod_col = st.selectbox("Select Product ID Column", cols)
    with c2:
        verb_col = st.selectbox("Select Verbatim Column", cols)
        
    # Process Text (only once using session state for speed)
    if st.button("Run NLP Analysis"):
        with st.spinner("Lemmatizing scent descriptions..."):
            df['cleaned'] = df[verb_col].apply(lambda x: clean_text(x, custom_stop_list))
            st.session_state['data'] = df

    if 'data' in st.session_state:
        df = st.session_state['data']
        tab1, tab2 = st.tabs(["Single Product Focus", "Comparison Mode"])
        
        with tab1:
            product_list = sorted(df[prod_col].unique())
            selected_prod = st.selectbox("Choose a Fragrance", product_list)
            prod_data = df[df[prod_col] == selected_prod]['cleaned']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### ☁️ Scent Word Cloud")
                fig_wc = generate_word_cloud(prod_data, palettes[selected_palette], min_freq)
                if fig_wc: st.pyplot(fig_wc)
                else: st.warning("Not enough data for this product.")
            
            with col_b:
                st.markdown("### 🌳 Scent Relationship Tree")
                fig_tree = generate_word_tree(prod_data, min_freq, palettes[selected_palette])
                if fig_tree: st.pyplot(fig_tree)
                else: st.warning("Not enough connections to form a tree.")
                
        with tab2:
            st.subheader("Head-to-Head Comparison")
            p_list = sorted(df[prod_col].unique())
            comp_col_1, comp_col_2 = st.columns(2)
            
            p1 = comp_col_1.selectbox("Product A", p_list, index=0)
            p2 = comp_col_2.selectbox("Product B", p_list, index=min(1, len(p_list)-1))
            
            data_a = df[df[prod_col] == p1]['cleaned']
            data_b = df[df[prod_col] == p2]['cleaned']
            
            comp_col_1.pyplot(generate_word_cloud(data_a, palettes[selected_palette], min_freq))
            comp_col_2.pyplot(generate_word_cloud(data_b, palettes[selected_palette], min_freq))
            
            comp_col_1.pyplot(generate_word_tree(data_a, min_freq, palettes[selected_palette]))
            comp_col_2.pyplot(generate_word_tree(data_b, min_freq, palettes[selected_palette]))

        # --- PPTX Export ---
        st.divider()
        if st.button("📦 Export Presentation"):
            prs = Presentation()
            
            def add_slide_to_pptx(title, fig):
                if fig is None: return
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                slide.shapes.title.text = title
                img_stream = io.BytesIO()
                fig.savefig(img_stream, format='png', bbox_inches='tight')
                slide.shapes.add_picture(img_stream, Inches(0.5), Inches(1.5), width=Inches(9))

            add_slide_to_pptx(f"Word Cloud: {selected_prod}", fig_wc)
            add_slide_to_pptx(f"Relationship Tree: {selected_prod}", fig_tree)
            
            binary_output = io.BytesIO()
            prs.save(binary_output)
            st.download_button(
                label="Download PPTX",
                data=binary_output.getvalue(),
                file_name=f"Fragrance_Report_{selected_prod}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
else:
    st.info("Please upload an Excel file containing consumer feedback to start the analysis.")
