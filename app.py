import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain
from pptx import Presentation
from pptx.util import Inches
import io

# --- 1. SETUP ---
st.set_page_config(page_title="Fragrance Verbatim Lab", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

nlp = load_nlp()

# Data for themes
FONT_OPTIONS = {"Modern Sans": "arial.ttf", "Classic Serif": "times.ttf", "Elegant": "georgia.ttf", "Luxury": "pala.ttf", "Soft Modern": "segoeui.ttf"}
PALETTES = {"Floral": "Pastel1", "Woody": "GnBu", "Fresh": "Blues", "Citrus": "YlOrRd", "Luxury Night": "Purples", "Professional": "tab10"}

# --- 2. SESSION STATE FOR STOPWORDS ---
# This allows you to edit the list live in the app
if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = "a, about, all, am, an, and, are, as, at, be, because, been, being, but, by, can, could, do, enough, feel, for, from, have, he, her, here, hers, herself, him, himself, his, how, i, if, in, it, its, itself, just, less, let, like, little, lot, make, me, more, my, myself, not, of, on, or, ought, our, ours, ourselves, product, real, she, should, so, that, the, their, theirs, them, themselves, there, these, they, think, this, those, to, too, until, very, we, what, when, where, which, while, who, whom, why, will, with, would, you, your, yours, yourself, yourselves, smell, remind, think, is, may, also, bit, go, put, out, into, quite, something, really, seem, evoke, above, after, again, against, any, before, below, between, both, cannot, did, does, doing, down, during, each, few, further, had, has, having, most, no, nor, off, once, only, other, over, own, same, some, such, than, then, through, under, up, was, were, therefore, order, say, none, kind, kinda, either, one, nothing, almost, anything, everything, find"

# --- 3. CORE PROCESSING ---

def get_cleaned_data(data, text_col):
    stop_list = set([x.strip().lower() for x in st.session_state.custom_stopwords.split(",")])
    cleaned_docs = []
    if nlp:
        for doc in nlp.pipe(data[text_col].astype(str), batch_size=50):
            tokens = [t.lemma_.lower() for t in doc if t.lemma_.lower() not in stop_list and t.is_alpha and len(t.text) > 2]
            cleaned_docs.append(" ".join(tokens))
    return cleaned_docs

def generate_network_tree(cleaned_text, palette_name):
    """Generates a Correlation Tree similar to the user's uploaded image."""
    G = nx.Graph()
    # Build co-occurrence
    for text in cleaned_text:
        words = list(set(text.split()))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                weight = G.get_edge_data(words[i], words[j], {'weight': 0})['weight'] + 1
                G.add_edge(words[i], words[j], weight=weight)
    
    if len(G.nodes) < 2: return None
    
    # Create Maximum Spanning Tree (The 'Tree' skeleton)
    T = nx.maximum_spanning_tree(G, weight='weight')
    partition = community_louvain.best_partition(T)
    
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    pos = nx.spring_layout(T, k=0.5, iterations=50)
    
    cmap = plt.get_cmap(PALETTES[palette_name])
    node_colors = [cmap(partition[node] % 10) for node in T.nodes()]
    
    nx.draw_networkx_edges(T, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(T, pos, node_size=100, node_color=node_colors)
    
    # Label scaling based on degree (importance)
    for node, (x, y) in pos.items():
        size = 8 + (T.degree(node) * 2)
        ax.text(x, y, node, fontsize=size, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.axis('off')
    return fig

# --- 4. USER INTERFACE ---

st.sidebar.title("🖌️ Design & NLP")
selected_font = st.sidebar.selectbox("Font Style", list(FONT_OPTIONS.keys()))
selected_palette = st.sidebar.selectbox("Color Palette", list(PALETTES.keys()))
use_tfidf = st.sidebar.checkbox("Apply TF-IDF Weighting", value=True)

tab_main, tab_compare, tab_stops = st.tabs(["📊 Analysis", "⚖️ Comparison", "🚫 Exclusion List"])

with tab_stops:
    st.subheader("Edit Words to Exclude")
    st.write("Add or remove words (separated by commas). These will be ignored in all clouds and trees.")
    st.session_state.custom_stopwords = st.text_area("Stopword List", st.session_state.custom_stopwords, height=300)

uploaded_file = st.file_uploader("Load Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    cols = df.columns.tolist()
    prod_col = st.sidebar.selectbox("Product Column", cols)
    text_col = st.sidebar.selectbox("Verbatim Column", cols)

    with tab_main:
        prod = st.selectbox("Select Product", df[prod_col].unique())
        if st.button(f"Generate Visuals for {prod}"):
            sub_df = df[df[prod_col] == prod]
            cleaned = get_cleaned_data(sub_df, text_col)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Word Cloud")
                # (Wordcloud logic here - reuse your existing function)
                # ... [Wordcloud Plotting] ...
            
            with col2:
                st.subheader("Word Correlation Tree")
                tree_fig = generate_network_tree(cleaned, selected_palette)
                if tree_fig: st.pyplot(tree_fig)
                else: st.warning("Not enough data for a tree.")