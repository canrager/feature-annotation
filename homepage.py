"""
Streamlit page for feature annotation
"""

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import streamlit.components.v1 as components
from utils import tokens_to_html_with_highlighting, tokens_to_html_with_scores
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import time



########################
# Initialize session
########################

st.set_page_config(layout="wide")

with st.spinner("Loading feature annotator..."):
    # User inputs
    COLUMNS = ["user_label", "user_recall", "user_interp_rating", "user_special_flag", "user_notes", "feature_idx", "feature_submodule_type", "feature_layer_idx", "feature_training_run_name"]
    st.session_state['inputs'] = st.session_state.get('inputs', defaultdict(list))

    # # User ID
    # if 'user_id' not in st.session_state:
    #     conn = st.connection("gsheets", type=GSheetsConnection)
    #     df = conn.read(
    #         worksheet="annotations",
    #         usecols=np.arange(len(COLUMNS)).tolist(),
    #     ).dropna()

    #     user_id = df['user_id'].iloc[-1] + 1
    #     st.session_state['user_id'] = int(user_id)

    # Progress bar
    st.session_state["progress_cnt"] = st.session_state.get("progress_cnt", 0)

    # Total number of features
    if 'n_features' not in st.session_state:
        with open('feature_contexts.json') as f:
            data = json.load(f)
            n_features = len(data)
            st.session_state['n_features'] = n_features

    # Close session if all features are annotated
    if st.session_state["progress_cnt"] == st.session_state["n_features"]:
        print("All features annotated")
        # Save to google sheets
        with st.spinner("Saving annotations..."):
            # Write to google sheets
            # Appending to google sheets is not supported by the library. Reading, appending and overwriting seems too risky for data loss.
            conn = st.connection("gsheets", type=GSheetsConnection)
            conn.create(
                worksheet=str(time.time()),
                data=pd.DataFrame(st.session_state['inputs'], columns=COLUMNS),
            )
            st.cache_data.clear()
        st.switch_page("pages/endpage.py")




########################
# Main page
########################

N_CONTEXTS_IN_EXPERIMENT = 256

# Welcome message
message = '''
Welcome to the feature annotator!

We trained dictionaries to map dense internal activations of the `pythia-70m-deduped` model to a sparse representation of 32k features. 
Now, you can explore the features! 
We'll show you examples of input contexts and next-token-predictions where where a given feature activates.
Please annotate this feature on the bottom of the page.

We will explain the dictionary learining process in more detail in our forthcoming paper.
Feel free to reach out to canrager@gmail.com if you have feedback or any questions right away.
'''

if st.session_state["progress_cnt"] == 0:
    st.info(message, icon="👋")

# Load data
with open('feature_contexts.json') as f:
    data = json.load(f)

# Display data
data = data[str(st.session_state["progress_cnt"])]
feat = data['feature']
st.header(f'Feature {feat["feature_idx"]} in {feat["submodule_type"]} layer {feat["layer_idx"]}')



info_mean_act = f'''
Top 10 (or less) tokens ranked by feature activation.
A deeply blue token indicates a high mean feature activation across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
We measure the feature activation at the sequence position of the token shown here.
'''
st.markdown(f'#### Top mean feature activation on token', help=info_mean_act)
txt = tokens_to_html_with_scores(data['top_mean_activations'], show_scores=False)
st.write(txt, unsafe_allow_html=True)

info_logprob = f'''
Top 10 (or less) tokens ranked by the logprob of the token prediction.
A deeply blue token indicates a high mean feature activation across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
We measure the feature activation at the sequence position *before* the predicted token.
'''
st.markdown(f'#### Top probablity diff for token prediction', help=info_logprob)
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
st.write(f'#### Contexts with top feature activations')
for i, (tokens, activations) in enumerate(data['top_contexts']):
    if max(activations) > 1e-4: # Only show contexts with non-zero activations
        txt = tokens_to_html_with_highlighting(tokens, activations)
        txt += "<hr/>"
        st.write(txt, unsafe_allow_html=True)




#########################
# User input
#########################
        
st.header('Feature annotation')
# st.expander("Feature annotation", expanded=True)

# Text input for feature label
label_input = st.text_input('Concise feature annotation:\nSummarize what the feature is about in 1-5 words.', key="label_input")

# Slider for rating recall
recall_options = np.arange(-1, 11) * 10
recall_options = [f'{x} %'  for x in recall_options]
recall_options[0] = "Please select recall"
recall_options[1] = "0 % (no true contexts match)"
recall_options[-1] = "100 % (perfect recall)"
recall_input = st.select_slider('Estimate the recall of your annotation:\nWhich fraction of all contexts the feature significantly activates on does your summary match?', options=recall_options, key="recall_input")
recall_input = int(recall_input.split(" ")[0])

# Slider for rating interpretability
interp_options = np.arange(-1, 11)
interp_options = [f'{x}'  for x in interp_options]
interp_options[0] = "Please select rating"
interp_options[1] = "0 (not interpretable)"
interp_options[-1] = "10 (very interpretable)"
interp_input = st.select_slider('Rate the interpretability of the feature:\nHow easy is it to understand what the feature is about?', options=interp_options, key="interp_input")
interp_input = int(interp_input.split(" ")[0])

# Expecially interesting
special_flag_input = st.checkbox('This feature is especially interesting.', key="special_flag_input")

# Text input for notes
notes_input = st.text_input('(Optional) Further notes on the feature', key="notes_input")

# Add button to submit
def submit():
    # Increment progress bar
    st.session_state["progress_cnt"] += 1
    new_input = [
        label_input, 
        recall_input,
        interp_input,
        special_flag_input,
        notes_input,
        feat["feature_idx"], 
        feat["submodule_type"], 
        feat["layer_idx"], 
        feat["training_run_name"]
    ]
    for i, col in enumerate(COLUMNS):
        st.session_state['inputs'][col].append(new_input[i])

    # Replace inputs with defaults
    st.session_state['label_input'] = ""
    st.session_state['rating_input'] = recall_options[0]
    st.session_state['notes_input'] = ""
    st.session_state['special_flag_input'] = False

    # scroll to top of page
    components.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)

st.button('Submit', on_click=submit)
    
# Progress bar
st.text("")
progress = st.progress(st.session_state["progress_cnt"] * (1/st.session_state['n_features']), "Annotation progress")

# Footer message
footer_message = f'''
Your annotations will be lost if you close this tab before completing all {st.session_state['n_features']} features.
Thanks for contributing :pray:

Feel free to reach out to canrager@gmail.com.
'''
st.write(footer_message)