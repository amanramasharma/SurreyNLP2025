import streamlit as st
from predict import predict_tags
from utils import get_label_color

import sys
print("🔥 Python in use:", sys.executable)

st.set_page_config(page_title="Biomedical Token Tagger", layout="centered")
st.title("Biomedical Token Classifier")
st.markdown("This model uses BiGRU + BioWordVec to label tokens in biomedical sentences.")

text = st.text_area("Enter your biomedical sentence below:", height=150)

if st.button("Predict"):
    if text.strip():
        output = predict_tags(text)
        st.markdown("### Predictions:")
        for token, tag in output:
            color = get_label_color(tag)
            st.markdown(f"<span style='background-color:{color}; padding:3px; border-radius:4px;'>{token} — {tag}</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence to tag.")