import streamlit as st
st.header(":star: You made it! :star:")
st.write("Thanks a lot for your help. You can close this tab now.")
spreadsheet_link = "https://docs.google.com/spreadsheets/d/1QU3ZAb1gqXXR7KYcI3LoqRQ16fAeRidWNBxXy_VLjiY/edit?usp=sharing"
st.write(f"See the annotation results in this spreadsheet: [{spreadsheet_link}]({spreadsheet_link})")

if st.button("Restart annotation"):
    st.session_state["progress_cnt"] = 0
    st.session_state['user_id'] += 1
    st.switch_page("homepage.py")