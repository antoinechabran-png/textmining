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
st.set_page_config(page_title="Fragrance Verbatim Lab", layout="wide")

# --- NLP Setup ---
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback for local environments where model isn't linked
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

def clean_text(text, custom_stopwords):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha 
              and not token.is_stop 
              and token.lemma_ not in custom_stopwords]
    return " ".join(tokens)

# --- Visualizations ---
def generate_word_cloud(text_series, palette, font_path, min_freq):
    combined_text = " ".join(text_series)
    if not combined_text.strip():
        return None
    
    wc = WordCloud(
        background_color="white",
        colormap=palette,
        font_path=font_path,
        min_word_length=2,
        width=800, height=400
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_word_tree(text_series, min_freq, palette_name):
    # Create Co-occurrence Matrix
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=min_freq)
    sparse_matrix = vectorizer.fit_transform(text_series)
    words = vectorizer.get_feature_names_out()
    
    # Adjacency matrix: (Words x Words)
    adj_matrix = (sparse_matrix.T * sparse_matrix)
    adj_matrix.setdiag(0)
    
    G = nx.from_scipy_sparse_array(adj_matrix)
    mapping = {i: word for i, word in enumerate(words)}
    G = nx.relabel_nodes(G, mapping)
    
    if len(G.nodes) == 0:
        return None

    # Community Detection
    partition = community_louvain.best_partition(G)
    
    # Maximum Spanning Tree to create "Tree" effect
    T = nx.maximum_spanning_tree(G)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(T, k=0.5, seed=42)
    
    # Map colors
    cmap = plt.get_cmap(palette_name)
    node_colors = [partition[node] for node in T.nodes()]
    
    nx.draw_networkx_nodes(T, pos, node_size=700, node_color=node_colors, cmap=cmap, alpha=0.8)
    nx.draw_networkx_labels(T, pos, font_size=10, font_family="sans-serif")
    nx.draw_networkx_edges(T, pos, alpha=0.3)
    
    plt.axis('off')
    return fig

# --- UI Sidebar ---
st.sidebar.header("🧪 Lab Settings")
uploaded_file = st.sidebar.file_uploader("Upload Fragrance Data (Excel)", type=["xlsx"])

custom_stopwords_input = st.sidebar.text_area(
    "Custom Stopwords (comma separated)", 
    "smell, product, think, fragrance, perfume, like"
)
custom_stopwords = [x.strip() for x in custom_stopwords_input.split(",")]

min_freq = st.sidebar.slider("Min Word Frequency", 2, 10, 3)

st.sidebar.subheader("🎨 Design Studio")
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
    
    col1, col2 = st.columns(2)
    with col1:
        prod_col = st.selectbox("Product ID Column", cols)
    with col2:
        verb_col = st.selectbox("Verbatim Column", cols)
        
    # Process Text
    df['cleaned'] = df[verb_col].apply(lambda x: clean_text(x, custom_stopwords))
    
    tab1, tab2 = st.tabs(["Single Product Analysis", "Comparison Lab"])
    
    with tab1:
        product_list = df[prod_col].unique()
        selected_prod = st.selectbox("Select Product", product_list)
        prod_data = df[df[prod_col] == selected_prod]['cleaned']
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Scent Word Cloud")
            fig_wc = generate_word_cloud(prod_data, palettes[selected_palette], None, min_freq)
            if fig_wc: st.pyplot(fig_wc)
        
        with c2:
            st.subheader("Scent Relationship Tree")
            fig_tree = generate_word_tree(prod_data, min_freq, palettes[selected_palette])
            if fig_tree: st.pyplot(fig_tree)
            
    with tab2:
        st.subheader("Side-by-Side Comparison")
        comp_col1, comp_col2 = st.columns(2)
        
        p1 = comp_col1.selectbox("Product A", product_list, index=0)
        p2 = comp_col2.selectbox("Product B", product_list, index=min(1, len(product_list)-1))
        
        data_a = df[df[prod_col] == p1]['cleaned']
        data_b = df[df[prod_col] == p2]['cleaned']
        
        comp_col1.pyplot(generate_word_cloud(data_a, palettes[selected_palette], None, min_freq))
        comp_col2.pyplot(generate_word_cloud(data_b, palettes[selected_palette], None, min_freq))
        
        comp_col1.pyplot(generate_word_tree(data_a, min_freq, palettes[selected_palette]))
        comp_col2.pyplot(generate_word_tree(data_b, min_freq, palettes[selected_palette]))

    # --- PPTX Export ---
    if st.button("Export Lab Results to PPTX"):
        prs = Presentation()
        
        def add_slide(title, fig):
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.title.text = title
            img_stream = io.BytesIO()
            fig.savefig(img_stream, format='png')
            slide.shapes.add_picture(img_stream, Inches(1), Inches(1.5), width=Inches(8))

        if fig_wc: add_slide(f"Word Cloud: {selected_prod}", fig_wc)
        if fig_tree: add_slide(f"Word Tree: {selected_prod}", fig_tree)
        
        binary_output = io.BytesIO()
        prs.save(binary_output)
        st.download_button(
            label="Download PowerPoint",
            data=binary_output.getvalue(),
            file_name="Fragrance_Analysis.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
else:
    st.info("👋 Welcome to the Fragrance Verbatim Lab. Please upload an Excel file to begin.")
