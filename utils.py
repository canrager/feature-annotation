import numpy as np

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


def tokens_to_html_with_highlighting(tokens, opacity_values, max_len=150, act_norm=1.0):
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
    opacity_values = opacity_values[-max_len:]

    for i, (token, opacity) in enumerate(zip(tokens, opacity_values)) if opacity_values is not None else tokens:
        # Convert white background color to RGBA format
        if opacity_values is None:
            background_color = "white"
            text_color = "black"
        else:
            normed_opacity = opacity / act_norm
            background_color = f"rgba(0, 0, 255, {normed_opacity})"
            light_val = 230
            dark_val = 40
            text_color = f"rgba({dark_val}, {dark_val}, {dark_val})" if normed_opacity < 0.5 else f"rgba({light_val}, {light_val}, {light_val})"
        txt += token
        if all([c in newline_tokens for c in token]):
            # replace all instances with ⏎
            token_rep = len(token) * "⏎"
            brs = "<br>" * len(token)
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; title: {opacity}, white-space: pre-wrap;" title="{np.round(opacity, 3)}">{token_rep}</span>{brs}'
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

            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;" title="{np.round(opacity, 3)}">{token}</span>'
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
    max_score = np.max(scores)
    for i, (token, score) in enumerate(zip(tokens, scores)):
        # Convert white background color to RGBA format
        normed_score = score / max_score
        background_color = f"rgba(0, 0, 255, {normed_score})"
        light_val = 230
        dark_val = 40
        text_color = f"rgba({dark_val}, {dark_val}, {dark_val})" if normed_score < 0.5 else f"rgba({light_val}, {light_val}, {light_val})"
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

        html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;" title="{np.round(score, 3)}">{token}</span>'
        if show_scores:
            html += f" {scores[i]:.2f}"
        if i < len(tokens) - 1:
            html += ", "
    return html
    
