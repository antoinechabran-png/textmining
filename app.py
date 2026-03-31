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
from collections import Counter

# --- 1. SETUP ---
st.set_page_config(page_title="Fragrance Lab", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

nlp = load_nlp()

# Themes
FONT_OPTIONS = {"Modern Sans": "arial.ttf", "Classic Serif": "times.ttf", "Elegant": "georgia.ttf", "Luxury": "pala.ttf"}
PALETTES = {"Floral": "Pastel1", "Woody": "GnBu", "Fresh": "Blues", "Citrus": "YlOrRd", "Luxury Night": "Purples", "Professional": "tab10"}

if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = "a, about, all, am, an, and, are, as, at, be, because, been, being, but, by, can, could, do, enough, feel, for, from, have, he, her, here, hers, herself, him, himself, his, how, i, if, in, it, its, itself, just, less, let, like, little, lot, make, me, more, my, myself, not, of, on, or, ought, our, ours, ourselves, product, real, she, should, so, that, the, their, theirs, them, themselves, there, these, they, think, this, those, to, too, until, very, we, what, when, where, which, while, who, whom, why, will, with, would, you, your, yours, yourself, yourselves, smell, remind, think, is, may, also, bit, go, put, out, into, quite, something, really, seem, evoke, find, everything, anything, almost"

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Analysis Settings")
min_freq = st.sidebar.slider("Minimum Word Quotations", 2, 10, 2)
selected_font = st.sidebar.selectbox("Font Style", list(FONT_OPTIONS.keys()))
selected_palette = st.sidebar.selectbox("Color Palette", list(PALETTES.keys()))
use_tfidf = st.sidebar.checkbox("Apply TF-IDF Weighting", value=True)

# --- 3. PROCESSING LOGIC ---

def get_filtered_words(data, text_col):
    stop_list = set([x.strip().lower() for x in st.session_state.custom_stopwords.split(",")])
    all_words = []
    cleaned_docs = []
    
    # Pre-process and Lemmatize
    for doc in nlp.pipe(data[text_col].astype(str), batch_size=50):
        tokens = [t.lemma_.lower() for t in doc if t.lemma_.lower() not in stop_list and t.is_alpha and len(t.text) > 2]
        cleaned_docs.append(tokens)
        all_words.extend(tokens)
    
    # Apply Frequency Filter
    counts = Counter(all_words)
    valid_words = {word for word, count in counts.items() if count >= min_freq}
    
    # Reconstruct sentences with only valid words
    final_docs = [" ".join([w for w in doc if w in valid_words]) for doc in cleaned_docs]
    final_docs = [d for d in final_docs if d.strip()] # Remove empty lines
    return final_docs, valid_words

def generate_visuals(cleaned_text, palette_name, font_name, prod_name):
    if not cleaned_text:
        return None, None

    # --- WORD CLOUD ---
    all_words = " ".join(cleaned_text).split()
    weights = Counter(all_words)
    
    G = nx.Graph()
    for text in cleaned_text:
        words = list(set(text.split()))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                G.add_edge(words[i], words[j], weight=G.get_edge_data(words[i], words[j], {'weight': 0})['weight'] + 1)
    
    partition = community_louvain.best_partition(G) if len(G.nodes) > 1 else {w: 0 for w in weights}
    cmap = plt.get_cmap(PALETTES[palette_name])
    
    def color_func(word, **kwargs):
        return "rgb(%d, %d, %d)" % tuple([int(x*255) for x in cmap(partition.get(word, 0) % 10)[:3]])

    wc = WordCloud(background_color="white", color_func=color_func, width=1000, height=600).generate_from_frequencies(weights)
    
    # --- WORD TREE (Network) ---
    fig_tree, ax = plt.subplots(figsize=(10, 7))
    if len(G.nodes) > 1:
        T = nx.maximum_spanning_tree(G, weight='weight')
        pos = nx.kamada_kawai_layout(T)
        nx.draw_networkx_edges(T, pos, alpha=0.2, edge_color='gray')
        for node, (x, y) in pos.items():
            ax.text(x, y, node, fontsize=9 + (T.degree(node)*2), ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color_func(node), lw=1))
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "Not enough connections for a tree", ha='center')
        
    return wc, fig_tree

# --- 4. UI TABS ---
tab_main, tab_stops = st.tabs(["📊 Analysis", "🚫 Exclusion Editor"])

with tab_stops:
    st.session_state.custom_stopwords = st.text_area("Edit words to ignore:", st.session_state.custom_stopwords, height=250)

uploaded_file = st.file_uploader("Load Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    prod_col = st.selectbox("Product Column", df.columns)
    text_col = st.selectbox("Verbatim Column", df.columns)
    
    with tab_main:
        prod = st.selectbox("Select Product", df[prod_col].unique())
        if st.button("Generate Visuals"):
            sub_df = df[df[prod_col] == prod]
            cleaned, valid_words = get_filtered_words(sub_df, text_col)
            
            if not valid_words:
                st.error(f"No words found with frequency >= {min_freq}. Try lowering the slider.")
            else:
                wc, tree_fig = generate_visuals(cleaned, selected_palette, selected_font, prod)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Word Cloud")
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc); ax_wc.axis("off")
                    st.pyplot(fig_wc)
                with col2:
                    st.subheader("Word Tree (MST)")
                    st.pyplot(tree_fig)