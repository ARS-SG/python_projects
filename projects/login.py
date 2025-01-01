import streamlit as st
from streamlit_option_menu import option_menu
with st.sidebar:
    st.title("Matrix")
    name=st.text_input("Enter your full name:")
    age=st.number_input("Enter your current age:",step=1)
    password=st.text_input("Enter the password:")

    submit_button=st.button("SUBMIT")
    if submit_button:
        if password=="admin":
            if name and age:
                st.success(f"Name={name}")
                st.success(f"Age={age}")
                st.success(f"Welcome, {name}")
            elif name:
                st.error("Age is not filled")
            elif age:
                st.error("Name is not filled")
            else:
                st.error("Neither fields are filled")
        else:
            st.error("Incorrect password")
