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

# --- 1. SETUP & NLP LOAD ---
st.set_page_config(page_title="Fragrance Verbatim Lab", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        # Fallback if the model wasn't downloaded correctly
        return None

nlp = load_nlp()

# --- 2. HELPER FUNCTIONS ---

def clean_and_lemmatize(text_list):
    """Cleans, removes stopwords, and lemmatizes fragrance verbatims."""
    if nlp is None:
        return [str(t).lower() for t in text_list]
    
    cleaned_docs = []
    for doc in nlp.pipe(text_list.astype(str), batch_size=50):
        # We keep 'not' to preserve sentiment meaning in fragrance tests
        tokens = [token.lemma_.lower() for token in doc 
                  if (not token.is_stop or token.lemma_ == "not") 
                  and not token.is_punct and len(token.text) > 2]
        cleaned_docs.append(" ".join(tokens))
    return cleaned_docs

def get_word_cloud_data(data, text_col, use_tfidf, color_theme):
    """Processes text and returns a WordCloud object + community colors."""
    cleaned_text = clean_and_lemmatize(data[text_col])
    
    # TF-IDF Weighting
    if use_tfidf:
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(cleaned_text)
        weights = dict(zip(vec.get_feature_names_out(), matrix.sum(axis=0).A1))
    else:
        # Simple Frequency
        all_words = " ".join(cleaned_text).split()
        weights = pd.Series(all_words).value_counts().to_dict()

    # Community Detection (Co-occurrence)
    G = nx.Graph()
    for text in cleaned_text:
        words = list(set(text.split()))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                G.add_edge(words[i], words[j], weight=G.get_edge_data(words[i], words[j], {'weight': 0})['weight'] + 1)
    
    partition = community_louvain.best_partition(G) if len(G.nodes) > 1 else {}
    cmap = plt.get_cmap(color_theme)
    
    def color_func(word, **kwargs):
        cluster = partition.get(word, 0)
        rgba = cmap(cluster % 10)
        return "rgb(%d, %d, %d)" % (rgba[0]*255, rgba[1]*255, rgba[2]*255)

    wc = WordCloud(background_color="white", color_func=color_func, width=1000, height=600).generate_from_frequencies(weights)
    return wc

def create_ppt(images_and_titles):
    """Generates a PPTX file from the generated word clouds."""
    prs = Presentation()
    for img_buf, title_text in images_and_titles:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = title_text
        img_buf.seek(0)
        slide.shapes.add_picture(img_buf, Inches(1), Inches(1.5), width=Inches(8))
    
    buf = io.BytesIO()
    prs.save(buf)
    buf.seek(0)
    return buf

# --- 3. USER INTERFACE (SIDEBAR) ---
st.sidebar.title("🎨 Customization")
color_theme = st.sidebar.selectbox("Color Palette", ["tab10", "Set3", "Pastel1", "Dark2"])
use_tfidf = st.sidebar.checkbox("Apply TF-IDF Weighting", value=True)

# --- 4. MAIN APP LOGIC ---
st.title("👃 Fragrance Consumer Test Analyzer")
st.write("Upload your Excel file to transform verbatims into visual insights.")

uploaded_file = st.file_uploader("Load Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    cols = df.columns.tolist()
    
    col_l, col_r = st.columns(2)
    with col_l:
        prod_col = st.selectbox("Select Product Code Column", cols)
    with col_r:
        text_col = st.selectbox("Select Verbatim Column", cols)

    tab1, tab2 = st.tabs(["Single Product Cloud", "Product Comparison"])

    with tab1:
        selected_prod = st.selectbox("Choose Product", df[prod_col].unique())
        if st.button(f"Generate Analysis for {selected_prod}"):
            sub_df = df[df[prod_col] == selected_prod]
            wc = get_word_cloud_data(sub_df, text_col, use_tfidf, color_theme)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
            # Export to PPT
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            ppt_data = create_ppt([(img_buf, f"Word Cloud: {selected_prod}")])
            st.download_button("📥 Download PPT", ppt_data, f"{selected_prod}_Report.pptx")

    with tab2:
        p1 = st.selectbox("Product A", df[prod_col].unique(), key="p1")
        p2 = st.selectbox("Product B", df[prod_col].unique(), key="p2")
        
        if st.button("Compare Products"):
            c1, c2 = st.columns(2)
            ppt_images = []
            
            for p, col in [(p1, c1), (p2, c2)]:
                with col:
                    st.subheader(p)
                    wc = get_word_cloud_data(df[df[prod_col] == p], text_col, use_tfidf, color_theme)
                    fig, ax = plt.subplots()
                    ax.imshow(wc)
                    ax.axis("off")
                    st.pyplot(fig)
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    ppt_images.append((buf, f"Comparison: {p}"))
            
            ppt_comp = create_ppt(ppt_images)
            st.download_button("📥 Download Comparison PPT", ppt_comp, "Comparison_Report.pptx")

else:
    st.info("Please upload an Excel file to begin.")