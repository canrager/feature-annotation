# %%
import os
import json
import numpy as np
from collections import Counter, defaultdict
from markupsafe import escape
import streamlit as st

class ClusterCfg:
  def __init__(self, score_type=None, pos_reduction=None, abs_scores=None):
    self.score_type = score_type
    self.pos_reduction = pos_reduction
    self.abs_scores = abs_scores
    self.dim_reduction = 'nosvd' # always no SVD

# filename = './data/contexts_pythia-70m-deduped_tloss0.03_ntok10000_skip512_npos10_mlp.json'
# context_y = json.loads(open(filename).read())
# y_global_idx = np.array(list(context_y.keys()), dtype=int)

# Load cluster results
# @st.cache_data
def load_cluster_results(_ccfg):
  clusters_dir = './data'
  clusters_filename = os.path.join(clusters_dir, f'clusters_{_ccfg.score_type}_{_ccfg.pos_reduction}_{_ccfg.dim_reduction}.json')
  clustering_results = json.loads(open(clusters_filename).read())
  cluster_totals = [ int(s) for s in list(clustering_results.keys()) ]
  return clustering_results, cluster_totals

def get_context(idx):
    """given idx in range(0, 10658635), return dataset sample
    and predicted token index within sample, in range(1, 1024)."""
    idx = str(idx)
    return context_y[idx]['context'], context_y[idx]['y']

def get_contexts(idxs):
    """given a list of idxs in range(0, 10658635), return list of dataset samples
    and predicted token indexes within samples, in range(1, 1024)."""
    contexts = [context_y[str(idx)]['context'] for idx in idxs]
    ys = [context_y[str(idx)]['y'] for idx in idxs]
    return contexts, ys

def convert_global_idxs_to_token_str(idxs):
    """given a list of global indexes, return "token\t(document_id: X, global_token_id: Y)" strings"""
    y = [context_y[str(idx)]['y'] for idx in idxs]
    doc_idxs = [context_y[str(idx)]['document_idx'] for idx in idxs]
    token_strs = [f'{y[i]}   (doc {doc_idxs[i]})' for i in range(len(idxs))]
    # token_strs = [context_y[str(idx)]['y'] + f'\t(in document {context_y[str(idx)]["document_idx"]} with global_token_idx {idx})' for idx in str(idxs)]
    return token_strs


def find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores=False):
    if abs_scores:
        abs_int = 1
    else:
        abs_int = 0
    num_y = len(y_global_idx)
    ones = np.ones(num_y)
    mask = clustering_results[str(n_total_clusters)][abs_int] == cluster_idx * ones
    idxs = y_global_idx[mask]
    return idxs

def return_token_occurrences_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores=False, token="y"):
    """given a cluster index, return a list of tuples of (token, count) for all unique tokens"""
    idxs = find_global_idxs_for_tokens_in_cluster(clustering_results, cluster_idx, n_total_clusters, abs_scores)
    if token == "y":
        token_strs = [context_y[str(idx)]['y'] for idx in idxs]
    elif type(token) == int:
        token_strs = [context_y[str(idx)]['context'][token] for idx in idxs]
    else:
        raise ValueError(f"token must be 'y' or an integer, not {token}")
    counts = Counter(token_strs)
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    cnt_dict = defaultdict(list)
    for y, count in counts:
        cnt_dict[count].append(y)
    return cnt_dict

def render_context_y(text, y=None, render_newlines=False):
    text = "".join(text)
    text = escape(text) # display html tags as text
    if y:
        context_formatted = f"<pre style='white-space:pre-wrap;'>{text}<span style='background-color: rgba(0, 255, 0, 0.5)'>{y}</span></pre>"
    else:
        context_formatted = f"<pre style='white-space:pre-wrap;'>{text}</pre>"
    if render_newlines:
        context_formatted = context_formatted.replace("\n", "<br>") # display newlines in html
    st.write(context_formatted, unsafe_allow_html=True)

def tokens_to_html(tokens, max_len=150):
    """Given a list of tokens (strings), returns html for displaying the tokenized text.
    """
    newline_tokens = ['\n', '\r', '\r\n', '\v', '\f']
    html = ""
    txt = ""
    if len(tokens) > max_len:
        html += '<span>...</span>'
    tokens = tokens[-max_len:]
    for i, token in enumerate(tokens):
        background_color = "white" if i != len(tokens) - 1 else "#FF9999"
        txt += token
        if all([c in newline_tokens for c in token]):
            # replace all instances with ⏎
            token_rep = len(token) * "⏎"
            brs = "<br>" * len(token)
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token_rep}</span>{brs}'
        else:
            # replace any $ with \$ to avoid markdown interpretation
            token = token.replace("$", "\$")
            # replace any < with &lt; to avoid html interpretation
            # token = token.replace("<", "&lt;")
            # replace any > with &gt; to avoid html interpretation
            # token = token.replace(">", "&gt;")
            # replace any & with &amp; to avoid html interpretation
            token = token.replace("&", "&amp;")
            # replace any _ with \_ to avoid markdown interpretation
            token = token.replace("_", "\_")
            # also escape * to avoid markdown interpretation
            token = token.replace("*", "\*")
            # there's also an issue with the backtick, so escape it
            token = token.replace("`", "\`")

            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token}</span>'
    if "</" in txt:
        return "CONTEXT NOT LOADED FOR SECURITY REASONS SINCE IT CONTAINS HTML CODE (could contain javascript)."
    else:
        return html


def tokens_to_html_with_highlighting(tokens, opacity_values=None, contains_scores=False, max_len=150):
    """
    Given a list of tokens (strings), returns html for displaying the tokenized text.
    No answer tokens.
    """
    newline_tokens = ['\n', '\r', '\r\n', '\v', '\f']
    html = ""
    txt = ""
    if len(tokens) > max_len:
        html += '<span>...</span>'
    tokens = tokens[-max_len:]
    if contains_scores:
        tokens = [tok[0] for tok in tokens]
        score_values = [tok[1] for tok in tokens]

    if opacity_values is not None:
        opacity_values = opacity_values[-max_len:]  # Ensure activation_values match the truncated tokens list
        opacity_values -= np.min(opacity_values)  # Shift activation_values to start from 0
        opacity_values /= np.max(opacity_values)  # Normalize activation_values to [0, 1]

    for i, token in enumerate(tokens):
        # Convert white background color to RGBA format
        if opacity_values is None:
            background_color = "white"
            text_color = "black"
        else:
            opacity = opacity_values[i]
            background_color = f"rgba(0, 0, 255, {opacity})"
            light_val = 230
            dark_val = 40
            text_color = f"rgba({dark_val}, {dark_val}, {dark_val})" if opacity < 0.5 else f"rgba({light_val}, {light_val}, {light_val})"
        txt += token
        if all([c in newline_tokens for c in token]):
            # replace all instances with ⏎
            token_rep = len(token) * "⏎"
            brs = "<br>" * len(token)
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;">{token_rep}</span>{brs}'
        else:
            # replace any $ with \$ to avoid markdown interpretation
            token = token.replace("$", "\$")
            # replace any < with &lt; to avoid html interpretation
            # token = token.replace("<", "&lt;")
            # replace any > with &gt; to avoid html interpretation
            # token = token.replace(">", "&gt;")
            # replace any & with &amp; to avoid html interpretation
            token = token.replace("&", "&amp;")
            # replace any _ with \_ to avoid markdown interpretation
            token = token.replace("_", "\_")
            # also escape * to avoid markdown interpretation
            token = token.replace("*", "\*")
            # there's also an issue with the backtick, so escape it
            token = token.replace("`", "\`")

            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;">{token}</span>'
    if "</" in txt:
        return "CONTEXT NOT LOADED FOR SECURITY REASONS SINCE IT CONTAINS HTML CODE (could contain javascript)."
    else:
        return html

def tokens_to_html_with_scores(tokens_and_scores, show_scores=True):
    """
    Output a string liting tokens and their scores.
    """
    newline_tokens = ['\n', '\r', '\r\n', '\v', '\f']
    html = ""
    tokens, scores = [], []
    for token, score in tokens_and_scores:
        if float(score) > 1e-4:
            tokens.append(token)
            scores.append(score)

    # Normalize scores to [, 1]
    scores_norm = np.array(scores)
    scores_norm /= np.max(scores_norm)
    for i, (token, score_norm) in enumerate(zip(tokens, scores_norm)):
        # Convert white background color to RGBA format
        background_color = f"rgba(0, 0, 255, {score_norm})"
        light_val = 230
        dark_val = 40
        text_color = f"rgba({dark_val}, {dark_val}, {dark_val})" if score_norm < 0.5 else f"rgba({light_val}, {light_val}, {light_val})"
        # replace any lewline character with ⏎
        for newline in newline_tokens:
            token = token.replace(newline, "⏎")
        # replace any $ with \$ to avoid markdown interpretation
        token = token.replace("$", "\$")
        # replace any < with &lt; to avoid html interpretation
        # token = token.replace("<", "&lt;")
        # replace any > with &gt; to avoid html interpretation
        # token = token.replace(">", "&gt;")
        # replace any & with &amp; to avoid html interpretation
        token = token.replace("&", "&amp;")
        # replace any _ with \_ to avoid markdown interpretation
        token = token.replace("_", "\_")
        # also escape * to avoid markdown interpretation
        token = token.replace("*", "\*")
        # there's also an issue with the backtick, so escape it
        token = token.replace("`", "\`")

        html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;">{token}</span>'
        if show_scores:
            html += f" {scores[i]:.2f}"
        if i < len(tokens) - 1:
            html += ", "
    return html
    
