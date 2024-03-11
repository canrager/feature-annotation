"""
Streamlit page for component annotation
"""

import streamlit as st
from streamlit_gsheets import GSheetsConnection
import streamlit.components.v1 as components
from utils import tokens_to_html
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import time



########################
# Initialize session
########################

N_CONTEXTS_IN_EXPERIMENT = 256
COLUMNS = ["user_label", "user_interp", "user_complexity", "user_notes", "component_set_name", "component_idx", "component_submodule_type", "component_layer_idx", "component_training_run_name"]
dataset_dir = "demo_contexts.json" #"sparse-dense_random-RC_contexts.json"

st.set_page_config(layout="wide")

with st.spinner("Loading component annotator..."):
    # User inputs
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

    # Load data
    if 'n_components' not in st.session_state:
        with open(dataset_dir) as f:
            st.session_state['data'] = json.load(f)
            n_components = len(st.session_state['data'])
            st.session_state['n_components'] = n_components

    def save_to_gsheets():
        with st.spinner("Saving annotations..."):
            # Write to google sheets
            # Appending to google sheets is not supported by the library. Reading, appending and overwriting seems too risky for data loss.
            conn = st.connection("gsheets", type=GSheetsConnection)
            conn.create(
                worksheet=str(time.time()),
                data=pd.DataFrame(st.session_state['inputs'], columns=COLUMNS),
            )
            st.cache_data.clear()
            st.session_state['inputs'] = defaultdict(list)

    # Close session if all components are annotated
    if st.session_state["progress_cnt"] == st.session_state["n_components"]:
        save_to_gsheets()
        st.switch_page("pages/endpage.py")
    elif st.session_state["progress_cnt"] % 10 == 0 and st.session_state["progress_cnt"] > 0:
        save_to_gsheets()





########################
# Main page
########################

# Welcome message
message = '''
Welcome to the model component annotator!

We gathered information on components of a neural network and prepared it for you to explore. 
Please help us by annotating how interpretable the components are. 
Watch the walkthrough video for this annotation tool here: https://youtu.be/Opo41GkVEok
'''

if st.session_state["progress_cnt"] == 0:
    st.info(message, icon="👋")


# Display data
data = st.session_state['data'][str(st.session_state["progress_cnt"])]
comp = data['component']
st.header(f'Component #{st.session_state["progress_cnt"]}')
st.write(f'')

info_pos_logprob = f'''
Top 10 (or less) tokens ranked by the logprob of the token prediction.
Hover over a token (and wait ~3s) to see the score.
A deeply blue token indicates a high mean logprob across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
We measure the component activation at the sequence position *before* the predicted token.
'''
st.markdown(f'##### Tokens most promoted by this component (mean)', help=info_pos_logprob)
tokens, scores = zip(*data['top_logprob_diff'])
txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
st.write(txt, unsafe_allow_html=True)


info_neg_logprob = f'''
Bottom 10 (or less) tokens ranked by the logprob of the token prediction.
Hover over a token (and wait ~3s) to see the score.
A deeply blue token indicates a high mean logprob across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
We measure the component activation at the sequence position *before* the predicted token.
'''
st.markdown(f'##### Tokens most suppressed by this component (mean)', help=info_neg_logprob)
tokens, scores = zip(*data['bottom_logprob_diff'])
txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
st.write(txt, unsafe_allow_html=True)

info_top_mean_act = f'''
Top 10 (or less) tokens ranked by component activation.
Hover over a token (and wait ~3s) to see the score.
A deeply blue token indicates a high mean component activation across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
We measure the component activation at the sequence position of the token shown here.
'''
st.markdown(f'##### Tokens which most stimulate this component (mean)', help=info_top_mean_act)
tokens, scores = zip(*data['top_mean_activations'])
txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
st.markdown(txt, unsafe_allow_html=True)

# info_bottom_mean_act = f'''
# Bottom 10 (or less) tokens ranked by component activation.
# Hover over a token (and wait ~3s) to see the score.
# A deeply blue token indicates a high mean component activation across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
# We measure the component activation at the sequence position of the token shown here.
# '''
# st.markdown(f'##### Tokens which most negatively stimulate this component (mean)', help=info_bottom_mean_act)
# tokens, scores = zip(*data['bottom_mean_activations'])
# txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
# if txt == "":
#     st.markdown('*No tokens found which have a significantly negative stimuation effect on this component.*')
# st.markdown(txt, unsafe_allow_html=True)

# st.write(f'#### Top logit diff for predicting:')
# tokens, scores = zip(*data['top_logit_diff'])
# txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
# st.write(txt, unsafe_allow_html=True)

# st.write(f'#### Top negative logit diff for predicting:')
# tokens, scores = zip(*data['bottom_logit_diff'])
# txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
# st.write(txt, unsafe_allow_html=True)

# Display first ten contexts from the dataset
info_context = f'''
Top 10 component activations with contexts.
Hover over a token (and wait ~3s) to see the score.
A deeply blue token indicates a high activation of the token in this specific contextr.
'''
st.markdown(f'##### Full input paragraphs of tokens which most stimulate this component', help=info_context)
# find max activation across all contexts
max_activations = [max(activations) for tokens, activations in data['top_contexts']]
global_max_activation = max(max_activations) # across all contexts shown for this component
for ((tokens, activations), max_act) in zip(data['top_contexts'], max_activations):
    activations = np.array(activations)
    if max_act < 1e-4: # Only show contexts with non-zero activations
        continue
    txt = tokens_to_html(tokens, activations, score_norm=global_max_activation)
    txt += "<hr/>" # Separator
    st.markdown(txt, unsafe_allow_html=True)




#########################
# User input
#########################
        
st.header('Component annotation')
# st.expander("component annotation", expanded=True)

# Text input for component label
label_input = st.text_input('**Concise component annotation:** Summarize what the component is about in 1-5 words.', key="label_input")

# Special component flag
# special_flag_input = st.checkbox('This component is especially interesting.', key="special_flag_input")

# Slider for rating interpretability
interp_options = np.arange(-1, 21) * 5
interp_options = [f'{x} %'  for x in interp_options]
interp_options[0] = "Please select"
interp_options[1] = "0 % (themes are conceptually disconnected)"
interp_options[-1] = "100 % (single common theme across contexts)"
interp_prompt = "**Interpretability score:** How coherent are the examples shown above? Does the component consistently activate on (or promote) the same concept or rather various distinct concepts?"
interp_input = st.select_slider(interp_prompt, options=interp_options, key="interp_input")
interp_input = interp_input.split(" ")[0]

# select_slider for rating semantic complexity
complexity_options = np.arange(-1, 21) * 5
complexity_options = [f'{x} %'  for x in complexity_options]
complexity_options[0] = "Please select"
complexity_options[1] = "0 % (always same token regardless of context)"
complexity_options[-1] = "100 % (diverse tokens in broad context)"
complexity_prompt = "**Semantic complexity score:** How broad is the concept the token fires on? Does the component activate on (or promote) diverse tokens of a general theme or simply the same token all over again?"
complexity_input = st.select_slider(complexity_prompt, options=complexity_options, key="complexity_input")
complexity_input = complexity_input.split(" ")[0]

# Slider for rating recall
# recall_options = np.arange(-1, 11) * 10
# recall_options = [f'{x} %'  for x in recall_options]
# recall_options[0] = "Please select recall"
# recall_options[1] = "0 % (no true contexts match)"
# recall_options[-1] = "100 % (perfect recall)"
# recall_input = st.select_slider('Estimate the recall of your annotation:\nWhich fraction of all contexts the component significantly activates on does your summary match?', options=recall_options, key="recall_input")
# recall_input = recall_input.split(" ")[0]

# Text input for notes
notes_input = st.text_input('(Optional) Further notes on the component', key="notes_input")

# Add button to submit
def submit():
    # Increment progress bar
    st.session_state["progress_cnt"] += 1
    new_input = [
        label_input, 
        interp_input,
        complexity_input,
        notes_input,
        comp["set_name"],
        comp["feature_idx"], 
        comp["submodule_type"], 
        comp["layer_idx"], 
        comp["training_run_name"]
    ]
    for i, col in enumerate(COLUMNS):
        st.session_state['inputs'][col].append(new_input[i])

    # Replace inputs with defaults
    st.session_state['label_input'] = ""
    # st.session_state['recall_input'] = recall_options[0]
    st.session_state['interp_input'] = interp_options[0]
    st.session_state['complexity_input'] = complexity_options[0]
    st.session_state['notes_input'] = ""
    st.session_state['special_flag_input'] = False

    # scroll to top of page
    components.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)

st.button('Submit', on_click=submit)
    
# Progress bar
st.text("")
progress = st.progress(st.session_state["progress_cnt"] * (1/st.session_state['n_components']), "Annotation progress")

# Footer message
st.write(f'''
Closing this tab resets the annotation session. We save your progress every 10 annotations.
Thanks for contributing :pray: 

Contact us at canrager@gmail.com.
''')