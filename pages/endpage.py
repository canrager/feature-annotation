import streamlit as st
st.header(":star: You made it! :star:")
st.write("Thanks a lot for your help. You can close this tab now.")
if st.button("Restart annotation"):
    st.session_state["progress_cnt"] = 0
    st.switch_page("homepage.py")