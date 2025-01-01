import streamlit as st
st.title("ELECTRICITY BILL")
st.subheader("CUSTOMER DETAILS")

name=st.text_input("enter customer name: ")
id=st.number_input("enter customer id: ")
units=st.number_input("enter units: ")

submit_button=st.button("SUBMIT")

if submit_button:   #if the button is clicked
    st.subheader("BILLING DETAILS")   #billing details section
    st.info(f"NAME: {name}")
    st.info(f"ID: {id}")
    if(units<=100):
        st.success("no changes")
    elif (100<units<=200):
        amount=(units-100)*5
        st.warning(f"total bill amount: {amount}")
    elif(units>200):
        amount=(units-100)*10
        st.error(f"total bill amount: {amount}")
else:    #if the submit button is not clicked
    st.error("CLICK SUBMIT!!")
