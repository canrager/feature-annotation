"""
Streamlit page for component annotation
"""

import streamlit as st
import streamlit.components.v1 as components
from utils import tokens_to_html
import json
import numpy as np
from collections import defaultdict
import time
from google.cloud import firestore
from google.oauth2 import service_account

########################
# Initialize session
########################

N_CONTEXTS_IN_EXPERIMENT = 256
TOTAL_ANNOTATIONS_PAID = 310
dataset_dir = "sparse-dense_random-RC_contexts.json"

st.set_page_config(layout="wide")
with st.spinner("Loading component annotator..."):

    # Connection to firebase
    fb_credentials = st.secrets["firebase"]
    creds = service_account.Credentials.from_service_account_info(fb_credentials)
    db = firestore.Client(credentials=creds, project="feature-annotation")

    # Find a random component with the least annotations
    st.session_state["progress_cnt"] = st.session_state.get("progress_cnt", 0)
    st.session_state["total_annotations"] = st.session_state.get("total_annotations", 0)
    st.session_state["sample_id"] = st.session_state.get("sample_id", None)

    if st.session_state["sample_id"] is None:
        stats = db.collection("stats").stream()
        cnt_dict = defaultdict(list)
        st.session_state["total_annotations"] = 0
        for stat in stats:
            cnt = stat.to_dict()["annotation_count"]
            st.session_state["total_annotations"] += cnt
            cnt_dict[cnt].append(stat.id)
        print(list(cnt_dict.keys()))
        least_annotated = min(cnt_dict.keys())
        sample_id = np.random.choice(cnt_dict[least_annotated])
        st.session_state['sample_id'] = sample_id

    # Load data
    if 'n_components' not in st.session_state:
        with open(dataset_dir) as f:
            st.session_state['data'] = json.load(f)
            n_components = len(st.session_state['data'])
            st.session_state['n_components'] = n_components
            
    # Close session if all components are annotated
    st.session_state["paid_mode"] = st.session_state.get("paid_mode", True)
    if st.session_state["paid_mode"] and st.session_state["total_annotations"] >= TOTAL_ANNOTATIONS_PAID:
        st.switch_page("pages/endpage.py")





########################
# Main page
########################

# Welcome message
message = '''
Welcome to the model component annotator!

We gathered information on components of a neural network and prepared it for you to explore. 
Please help us by annotating how interpretable the components are. 
Watch the walkthrough video for this annotation tool here: https://youtu.be/RioPfQyZqBI
'''
if st.session_state["progress_cnt"] == 0:
    st.info(message, icon="👋")


# Display data
data = st.session_state['data'][str(st.session_state["sample_id"])]
comp = data['component']
st.header(f'Component #{st.session_state["sample_id"]}')
if st.session_state["paid_mode"]:
    st.write(f'We will reimburse you for your time 30$/hr (please track approximately how long you spend on this). We appreciate your help!')
else:
    st.write(f'You are annotating voluntarily. We appreciate your help!')
st.write(f'')

info_pos_logprob = f'''
Top 10 (or less) tokens ranked by the counterfactual log-prob difference of the predicted token.
(The counterfactual log-prob difference is the difference between the log-prob in a clean fwd pass 
and the log-prob of a fwd pass where we set the output of the given component to zero.)
Hover over a token (and wait ~3s) to see the score.
A deeply blue token indicates a high mean logprob across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
We measure the counterfactual log-prob diff at the sequence position *before* the predicted token.
'''
st.markdown(f'##### Tokens most promoted by this component (mean)', help=info_pos_logprob)
tokens, scores = zip(*data['top_logprob_diff'])
txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
st.write(txt, unsafe_allow_html=True)

# info_neg_logprob = f'''
# Bottom 10 (or less) tokens ranked by the logprob of the token prediction.
# Hover over a token (and wait ~3s) to see the score.
# A deeply blue token indicates a high mean logprob across {N_CONTEXTS_IN_EXPERIMENT} random contexts.
# We measure the component activation at the sequence position *before* the predicted token.
# '''
# st.markdown(f'##### Tokens most suppressed by this component (mean)', help=info_neg_logprob)
# tokens, scores = zip(*data['bottom_logprob_diff'])
# txt = tokens_to_html(tokens, scores, comma_separate_tokens=True, render_newlines=False, score_threshold=1e-5)
# st.write(txt, unsafe_allow_html=True)

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

# Optional: Text input for notes
notes_input = st.text_input('(Optional) Further notes on the component', key="notes_input")

# Optional: Your user name
username_input = st.text_input('(Optional) Your user name', key="username_input")

# Add button to submit
def submit():
    # Increment progress bar
    st.session_state["progress_cnt"] += 1

    # Save inputs to firebase
    annotation_title = "no" + str(st.session_state["total_annotations"]) + f"_at{time.time()}"

    annotation_input = dict(
        sample_id=st.session_state["sample_id"],
        component_set_name=comp["set_name"],
        component_idx=comp["feature_idx"],
        component_submodule_type=comp["submodule_type"],
        component_layer_idx=comp["layer_idx"],
        component_training_run_name=comp["training_run_name"],
        user_label=label_input,
        user_interp=interp_input,
        user_complexity=complexity_input,
        user_notes=notes_input,
        user_name=username_input,
    )
    annotation_doc_ref = db.collection("annotations").document(annotation_title)
    annotation_doc_ref.set(annotation_input)

    # Update annotation count in stats
    stat_ref = db.collection("stats").document(str(st.session_state["sample_id"]))
    stat = stat_ref.get().to_dict()
    stat["annotation_count"] += 1
    stat_ref.set(stat)

    # Replace input fields with default values
    st.session_state['label_input'] = ""
    # st.session_state['recall_input'] = recall_options[0]
    st.session_state['interp_input'] = interp_options[0]
    st.session_state['complexity_input'] = complexity_options[0]
    st.session_state['notes_input'] = ""
    st.session_state['special_flag_input'] = False

    st.session_state['sample_id'] = None

    # scroll to top of page
    components.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>", height=0)

st.button('Submit', on_click=submit)
    

# Footer
st.text("")
annotations_left = TOTAL_ANNOTATIONS_PAID - st.session_state["total_annotations"]
st.write(f'''
You annotated {st.session_state["progress_cnt"]} components in this session. 
We need {annotations_left} more paid annotations to reach our goal.
Thanks for contributing :pray: 

Contact us at canrager@gmail.com.
''')