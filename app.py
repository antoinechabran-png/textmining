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

# --- 1. THE BULLETPROOF LOADER ---
@st.cache_resource
def load_nlp():
    # Attempt to load the model installed via requirements.txt
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # Final fallback: try to load by string name
        try:
            import en_core_web_sm
            return en_core_web_sm.load()
        except ImportError:
            return None

nlp = load_nlp()

# --- 2. APP CONFIG & SESSION STATE ---
st.set_page_config(page_title="Fragrance Lab", layout="wide")

if 'custom_stops' not in st.session_state:
    st.session_state.custom_stops = "a, about, all, am, an, and, are, as, at, be, because, been, being, but, by, can, could, do, enough, feel, for, from, have, he, her, here, hers, herself, him, himself, his, how, i, if, in, it, its, itself, just, less, let, like, little, lot, make, me, more, my, myself, not, of, on, or, ought, our, ours, ourselves, product, real, she, should, so, that, the, their, theirs, them, themselves, there, these, they, think, this, those, to, too, until, very, we, what, when, where, which, while, who, whom, why, will, with, would, you, your, yours, yourself, yourselves, smell, remind, think, is, may, also, bit, go, put, out, into, quite, something, really, seem, evoke, find, everything, anything, almost, therefore, order, say, none, kind, kinda, either, one, nothing"

# --- 3. PROCESSING FUNCTIONS ---

def get_filtered_data(data, text_col, min_freq):
    # Check if nlp loaded correctly to avoid AttributeError
    if nlp is None:
        st.error("🚨 SpaCy model 'en_core_web_sm' not found. Please ensure your requirements.txt includes the download link.")
        return None, None

    stop_list = set([x.strip().lower() for x in st.session_state.custom_stops.split(",")])
    all_tokens = []
    docs_tokens = []
    
    # Process text
    texts = data[text_col].astype(str).tolist()
    for doc in nlp.pipe(texts, batch_size=50):
        tokens = [t.lemma_.lower() for t in doc if t.lemma_.lower() not in stop_list and t.is_alpha and len(t.text) > 2]
        docs_tokens.append(tokens)
        all_tokens.extend(tokens)
    
    counts = Counter(all_tokens)
    valid_words = {word for word, count in counts.items() if count >= min_freq}
    
    final_docs = [" ".join([w for w in doc if w in valid_words]) for doc in docs_tokens]
    final_docs = [d for d in final_docs if d.strip()]
    
    return final_docs, valid_words

def create_visuals(cleaned_text, palette):
    if not cleaned_text: return None, None
    
    weights = Counter(" ".join(cleaned_text).split())
    G = nx.Graph()
    for text in cleaned_text:
        words = list(set(text.split()))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                G.add_edge(words[i], words[j], weight=G.get_edge_data(words[i], words[j], {'weight': 0})['weight'] + 1)
    
    partition = community_louvain.best_partition(G) if len(G.nodes) > 1 else {w: 0 for w in weights}
    cmap = plt.get_cmap(palette)
    
    def color_func(word, **kwargs):
        rgb = [int(x*255) for x in cmap(partition.get(word, 0) % 10)[:3]]
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

    wc = WordCloud(background_color="white", color_func=color_func, width=1000, height=600).generate_from_frequencies(weights)
    
    fig_tree, ax = plt.subplots(figsize=(10, 8))
    if len(G.nodes) > 1:
        T = nx.maximum_spanning_tree(G, weight='weight')
        pos = nx.kamada_kawai_layout(T)
        nx.draw_networkx_edges(T, pos, alpha=0.2, edge_color='gray')
        for node, (x, y) in pos.items():
            ax.text(x, y, node, fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color_func(node), boxstyle='round'))
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "Not enough connections for a tree.", ha='center'); ax.axis('off')
        
    return wc, fig_tree

# --- 4. UI ---

st.sidebar.title("Settings")
min_f = st.sidebar.slider("Min. Frequency", 2, 10, 2)
pal = st.sidebar.selectbox("Palette", ["Pastel1", "GnBu", "Blues", "YlOrRd", "tab10"])

t1, t2 = st.tabs(["Analysis", "Exclusions"])

with t2:
    st.session_state.custom_stops = st.text_area("Stopwords", st.session_state.custom_stops, height=300)

up = st.file_uploader("Upload Excel", type=["xlsx"])

if up:
    df = pd.read_excel(up)
    prod_c = st.selectbox("Product Column", df.columns)
    text_c = st.selectbox("Verbatim Column", df.columns)
    
    prod_val = st.selectbox("Select Product", df[prod_c].unique())
    
    if st.button("Generate"):
        sub = df[df[prod_c] == prod_val]
        res = get_filtered_data(sub, text_c, min_f)
        
        if res and res[0]:
            cleaned, valid = res
            wc, tree = create_visuals(cleaned, pal)
            
            c1, c2 = st.columns(2)
            with c1:
                f1, a1 = plt.subplots(); a1.imshow(wc); a1.axis("off"); st.pyplot(f1)
            with c2:
                st.pyplot(tree)
        else:
            st.warning("No words met the criteria. Check the exclusion list or decrease Min. Frequency.")