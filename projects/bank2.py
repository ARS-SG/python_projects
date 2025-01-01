# import the streamlit library
import streamlit as st

# give a title to our app
st.title('Welcome to Bank')

# TAKE WEIGHT INPUT in kgs
bal = st.number_input("Enter your balance")

# TAKE HEIGHT INPUT
# radio button to choose height format
status = st.radio('Select your height format: ',
				('deposit', 'withdraw'))

# compare status value
if(status == 'deposit'):
	dep = st.number_input("enter deposit amount")

	try:
		bal=bal+dep
	except:
		st.text("Enter some value of deposit")

else:
	wd = st.number_input('enter withdraw amount')

	try:
		bal=bal-wd
	except:
		st.text("Enter some value of withdrawal")


# check if the button is pressed or not
if(st.button('Calculate Balance')):

	st.text("Your balance is {}.".format(bal))