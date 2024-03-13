import streamlit as st

st.header(":star: We made it! :star:")
st.write("We gathered all annotations we need. Feel free to continue annotating, we ran out of money to pay you though. :sob: :money_with_wings: :money_with_wings: :money_with_wings. Thanks again for your help!")

if st.button("Continue annotating"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["paid_mode"] = False
    st.switch_page("homepage.py")