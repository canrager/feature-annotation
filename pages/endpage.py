import streamlit as st

st.header(":star: We made it! :star:")
st.write("We gathered the minimum amout of annotations. Feel free to explore our tool and continue annotating voluntarily. Thanks for your help!")

if st.button("Continue annotating"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["paid_mode"] = False
    st.switch_page("homepage.py")