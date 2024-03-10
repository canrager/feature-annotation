import numpy as np

def tokens_to_html(tokens, scores, max_len=150, score_norm=None, score_threshold=None, comma_separate_tokens=False, 
                   show_scores=False, render_newlines=True):
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
    scores = scores[-max_len:]

    if score_norm is None:
        score_norm = np.max(np.abs(scores))

    for i, (token, score) in enumerate(zip(tokens, scores)):
        if score_threshold:
            if np.abs(score) < score_threshold:
                continue
        # Highlighting
        normed_score = np.abs(score) / score_norm
        if score < 0:
            background_color = f"rgba(255, 0, 0, {normed_score})"
        else:
            background_color = f"rgba(0, 0, 255, {normed_score})"

        light_val = 230
        dark_val = 40
        text_color = f"rgba({dark_val}, {dark_val}, {dark_val})" if normed_score < 0.5 else f"rgba({light_val}, {light_val}, {light_val})"
        txt += token
        
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
        for newline in newline_tokens:
            token = token.replace(newline, "⏎")

        html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; color: {text_color}; white-space: pre-wrap;" title="{np.round(score, 3)}">{token}</span>'
        if render_newlines and all([c in newline_tokens for c in token]):
            brs = "<br>" * len(token)
            html += brs
        if show_scores:
            html += f" {scores[i]:.2f}"
        if comma_separate_tokens and i < len(tokens) - 1:
            html += ", "
    if "</" in txt and not comma_separate_tokens:
        return "CONTEXT NOT LOADED FOR SECURITY REASONS SINCE IT CONTAINS HTML CODE (could contain javascript)."
    else:
        return html