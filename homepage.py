"""
Streamlit page for feature annotation
"""

import streamlit as st
from utils import tokens_to_html_with_highlighting, tokens_to_html_with_scores
import json
import numpy as np

# Initialize counter
st.session_state["progress_cnt"] = st.session_state.get("progress_cnt", 0)

# Load data
with open('feature_contexts.json') as f:
    data = json.load(f)
    N_FEATURES = len(data)
    if st.session_state["progress_cnt"] == N_FEATURES:
        print("All features annotated")
        st.switch_page("pages/endpage.py")
    data = data[str(st.session_state["progress_cnt"])]


# Main page
feat = data['feature']
st.header(f'Feature {feat["feature_idx"]} in {feat["submodule_type"]} layer {feat["layer_idx"]}')
if st.session_state["progress_cnt"] == 0:
    st.write("Please annotate this feature using the sidebar on the left.")

st.write(f'#### Top mean feature activation on token')
txt = tokens_to_html_with_scores(data['top_mean_activations'], show_scores=False)
st.write(txt, unsafe_allow_html=True)

st.write(f'#### Top probablity diff for token prediction')
txt = tokens_to_html_with_scores(data['top_logprob_diff'], show_scores=False)
st.write(txt, unsafe_allow_html=True)

# st.write(f'#### Top negative probablity diff for predicting:')
# txt = tokens_to_html_with_scores(data['bottom_logprob_diff'], show_scores=False)
# st.write(txt, unsafe_allow_html=True)

# st.write(f'#### Top logit diff for predicting:')
# txt = tokens_to_html_with_scores(data['top_logit_diff'], show_scores=False)
# st.write(txt, unsafe_allow_html=True)

# st.write(f'#### Top negative logit diff for predicting:')
# txt = tokens_to_html_with_scores(data['bottom_logit_diff'], show_scores=False)
# st.write(txt, unsafe_allow_html=True)

# Display first ten contexts from the dataset
# st.write(f'#### Top feature activations with context')
for i, (tokens, activations) in enumerate(data['top_contexts']):
    st.write(f'#### Context {i+1}')
    txt = tokens_to_html_with_highlighting(tokens, activations)
    st.write(txt, unsafe_allow_html=True)
    st.write("-----------------------------------------------------------")


# Sidebar for user input
st.sidebar.header('Feature annotation')
st.sidebar.expander("Feature annotation", expanded=True)

feature_summary = st.sidebar.text_input('Summarize the activated tokens in 1-5 words', '')

# Radio for rating interpretability
radio_options = np.arange(1, 11)
rating = st.sidebar.radio('Interpretability rating', radio_options, 5)

# Add button to submit
if st.sidebar.button('Submit'):
    # Save feature annotation to file
    # Increment progress bar
    st.session_state["progress_cnt"] += 1
    annotation = {str(st.session_state["progress_cnt"]): dict(feature_summary=feature_summary, rating=str(rating))}
    with open("feature_annotations.json", "a") as f:
        json.dump(annotation, f)
    st.rerun()

# Progress bar
st.sidebar.text("")
progress = st.sidebar.progress(st.session_state["progress_cnt"] * (1/N_FEATURES), "Annotation progress")
# progress.progress(st.session_state["progress_cnt"] * (1/N_FEATURES), "Annotation progress")