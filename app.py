import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
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
st.set_page_config(page_title="Fragrance Verbatim Lab", layout="wide", page_icon="🌿")

# --- Lightweight NLP Setup (NLTK) ---
@st.cache_resource
def setup_nltk():
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, nltk_stop_words = setup_nltk()

def clean_text_light(text, custom_stopwords):
    if not text or pd.isna(text):
        return ""
    
    # Fast regex tokenization
    words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
    
    # Filter and Lemmatize
    cleaned = [lemmatizer.lemmatize(w) for w in words 
               if w not in nltk_stop_words 
               and w not in custom_stopwords]
    
    return " ".join(cleaned)

# --- Visualization Engines ---
def generate_word_cloud(text_series, palette):
    combined_text = " ".join(text_series)
    if not combined_text.strip():
        return None
    
    wc = WordCloud(
        background_color="white",
        colormap=palette,
        width=800, height=450,
        max_words=100
    ).generate(combined_text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def generate_word_tree(text_series, min_freq, palette_name):
    # Filter out very short responses
    valid_text = [t for t in text_series if len(t.split()) > 1]
    if not valid_text:
        return None

    try:
        vectorizer = CountVectorizer(min_df=min_freq)
        sparse_matrix = vectorizer.fit_transform(valid_text)
        words = vectorizer.get_feature_names_out()
        
        # Co-occurrence Logic
        adj_matrix = (sparse_matrix.T * sparse_matrix)
        adj_matrix.setdiag(0)
        
        G = nx.from_scipy_sparse_array(adj_matrix)
        G = nx.relabel_nodes(G, {i: word for i, word in enumerate(words)})
        
        if len(G.nodes) < 2: return None

        # Communities & Tree Structure
        partition = community_louvain.best_partition(G)
        T = nx.maximum_spanning_tree(G)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.kamada_kawai_layout(T)
        
        cmap = plt.get_cmap(palette_name)
        node_colors = [partition[node] for node in T.nodes()]
        
        nx.draw_networkx_nodes(T, pos, node_size=1200, node_color=node_colors, cmap=cmap, alpha=0.9)
        nx.draw_networkx_labels(T, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(T, pos, alpha=0.2)
        
        plt.axis('off')
        return fig
    except:
        return None

# --- App UI ---
st.title("🌿 Fragrance Verbatim Lab (Light Edition)")
st.markdown("Analyze scent descriptions instantly without heavy NLP models.")

with st.sidebar:
    st.header("📥 Data Input")
    uploaded_file = st.file_uploader("Upload Consumer Feedback (.xlsx)", type=["xlsx"])
    
    st.header("🛠️ Filters")
    custom_input = st.text_area("Custom Stopwords", "smell, product, fragrance, perfume, think, feel")
    custom_stops = set([x.strip().lower() for x in custom_input.split(",")])
    
    min_freq = st.slider("Min Word Frequency (for Tree)", 2, 10, 3)
    
    st.header("🎨 Palette")
    palettes = {"Citrus": "YlOrBr", "Woody": "copper", "Fresh": "GnBu", "Floral": "RdPu", "Pastel": "Pastel1"}
    selected_palette = st.selectbox("Theme", list(palettes.keys()))

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    cols = df.columns.tolist()
    
    c1, c2 = st.columns(2)
    prod_col = c1.selectbox("Product ID Column", cols)
    verb_col = c2.selectbox("Verbatim/Feedback Column", cols)

    if st.button("🚀 Analyze Data"):
        with st.spinner("Processing text..."):
            df['cleaned'] = df[verb_col].apply(lambda x: clean_text_light(x, custom_stops))
            st.session_state['df'] = df

    if 'df' in st.session_state:
        df = st.session_state['df']
        tab1, tab2 = st.tabs(["Single Product Analysis", "Comparison Lab"])
        
        with tab1:
            prod_id = st.selectbox("Select Product", sorted(df[prod_col].unique()))
            data = df[df[prod_col] == prod_id]['cleaned']
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Word Cloud")
                fig_wc = generate_word_cloud(data, palettes[selected_palette])
                if fig_wc: st.pyplot(fig_wc)
            with col_right:
                st.subheader("Scent Tree")
                fig_tree = generate_word_tree(data, min_freq, palettes[selected_palette])
                if fig_tree: st.pyplot(fig_tree)

        with tab2:
            p_list = sorted(df[prod_col].unique())
            cc1, cc2 = st.columns(2)
            p1 = cc1.selectbox("Product A", p_list, index=0)
            p2 = cc2.selectbox("Product B", p_list, index=min(1, len(p_list)-1))
            
            cc1.pyplot(generate_word_cloud(df[df[prod_col]==p1]['cleaned'], palettes[selected_palette]))
            cc2.pyplot(generate_word_cloud(df[df[prod_col]==p2]['cleaned'], palettes[selected_palette]))
            
            cc1.pyplot(generate_word_tree(df[df[prod_col]==p1]['cleaned'], min_freq, palettes[selected_palette]))
            cc2.pyplot(generate_word_tree(df[df[prod_col]==p2]['cleaned'], min_freq, palettes[selected_palette]))

        # --- PPTX Export ---
        if st.button("💾 Export Results to PPTX"):
            prs = Presentation()
            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.title.text = f"Analysis: {prod_id}"
            
            img_buffer = io.BytesIO()
            fig_wc.savefig(img_buffer, format='png')
            slide.shapes.add_picture(img_buffer, Inches(1), Inches(2), width=Inches(8))
            
            output = io.BytesIO()
            prs.save(output)
            st.download_button("Download PowerPoint", output.getvalue(), "Fragrance_Report.pptx")
else:
    st.info("Upload your fragrance feedback file to begin.")
