import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import mysql.connector
import re


myconn = mysql.connector.connect(host="localhost", user="root", passwd="", database="shopping")   #initialising the mysql connector
cur = myconn.cursor()           #creating the cursor
price=0                #initialising the price variable
st.title("ONLINE MINI SHOPPING CART")       #title

with st.sidebar:                           #creating the sidebar
    img = Image.open('cart.png')  # loading the static
    st.image(img,width=100)
    option=option_menu("CATEGORIES",options=["CLOTHING","ELECTRONICS","VEGETABLES","GROCERY"])           #options in the sidebar
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------

if option=="CLOTHING":
    st.subheader("CLOTHING ITEMS")
    col1, col2, col3 = st.columns(3)
    with col1:  # adding content to col1
        img1 = Image.open('jacket.jpg')  # loading the static
        st.image(img1,caption="Jacket")  # displaying static on webpage
        st.info("Price=100")
    with col2:  # adding content to col2
        img2 = Image.open('hat.jpg')  # loading the static
        st.image(img2,caption="Hat")  # displaying static on webpage
        st.info("Price=400")
    with col3:  # adding content to col3
        img3 = Image.open('belt.jfif')  # loading the static
        st.image(img3,caption="Belt")  # displaying static on webpage
        st.info("Price=500")


    s_title=st.text("Choose")
    jacket = st.checkbox('Jacket')
    hat=st.checkbox('Hat')
    belt=st.checkbox('Belt')
    cname = st.text_input("Enter customer name: ")
    age=st.slider("Enter your age:",min_value=1,max_value=100)
    mobile = st.text_input("Enter mobile number: ")
    email=st.text_input("Enter email: ")
    address=st.text_input("Enter address: ")
    coupon=st.selectbox("Would you like to use coupon and deduct 40 units",options=["No","Yes"])

# determining the total price in accordance to the checkboxes ticked
    if jacket and hat:
        price = 500
    elif jacket and belt:
        price = 600
    elif hat and belt:
        price = 900
    elif jacket:
        price=100
    elif hat:
        price=400
    else:
        price=500

    discount_price = price

    if (coupon == "Yes"):
        cwrite=st.text_input("Use coupon code 'FREE' and hit enter ")
        if cwrite=='FREE':
            discount_price = price - 40                       #if user accurately enters "FREE", the total price will be deducted by 40 units
            st.success("CONGRATULATIONS! 40 UNITS HAVE BEEN DEDUCTED FROM YOUR CART! ")
            st.info(f"Final Price after Coupon: {discount_price}")     #final total price displayed
        elif cwrite=='':
            st.warning("Please write the code")           #if nothing is written, this message will pop up
        else:
            st.error("Incorrect Coupon code.Try again.")      #if the coupon code is incorrect, this message will pop up
    else:
        discount_price=price


    def are_fields_filled(cname, age, mobile, email, address):
        return cname != "" and email != "" and age > 0 and mobile !="" and address !=""            #as long as these parameters are met

    submit_button=st.button("SUBMIT THE CART")

    if submit_button:
        pattern="^91"
        test=mobile         # the mobile number must begin with 91 in order for the rest of the code to work
        result = re.match(pattern, test)
        if result:
            def shop_cloth():
                sql = "insert into clothing(cname,age,mobile,email,address,price) values (%s,%s,%s,%s,%s,%s)"
                val = (cname, age, mobile, email, address, discount_price)
                try:
                    if are_fields_filled(cname, age, mobile, email, address):
                        # inserting the values into the table
                        cur.execute(sql, val)
                        myconn.commit()
                        # committed the transaction
                        st.success("Transaction has been saved")
                        st.info(f"{cur.rowcount} record inserted")
                        cur.execute("SELECT * FROM clothing WHERE cname = %s", (cname,))
                        records = cur.fetchall()
                        if records:
                            st.subheader("Recently Added Record:")
                            st.table(records)                        #displaying the recently added record in a table format
                    else:
                        st.warning("Please fill in all the required fields properly.")
                except Exception as e:
                    st.error(f"cannot process {e}")
                    myconn.rollback()


            shop_cloth()
        else:
            st.error("Enter mobile number properly")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------

if option=="ELECTRONICS":
    st.subheader("ELECTRONICS ITEMS")
    col1, col2, col3 = st.columns(3)
    with col1:  # adding content to col1
        img1 = Image.open('charger.jpg')  # loading the static
        st.image(img1,caption="Wired Charger")  # displaying static on webpage
        st.info("Price=1000")
    with col2:  # adding content to col2
        img2 = Image.open('portable.jpg')  # loading the static
        st.image(img2,caption="Portable Charger")  # displaying static on webpage
        st.info("Price=6700")
    with col3:  # adding content to col3
        img3 = Image.open('adapter.jpg')  # loading the static
        st.image(img3,caption="Adapter")  # displaying static on webpage
        st.info("Price=9000")

    s_title = st.text("Choose")
    wc = st.checkbox('Wired Charger')
    pc=st.checkbox('Portable Charger')
    ad=st.checkbox('Adapter')
    cname = st.text_input("Enter customer name: ")
    age=st.slider("Enter your age:",min_value=1,max_value=100)
    mobile = st.text_input("Enter mobile number: ")
    email=st.text_input("Enter email: ")
    address=st.text_input("Enter address: ")
    coupon = st.selectbox("Would you like to use coupon and deduct 40 units", options=["No", "Yes"])

    if wc and pc:
        price = 1000+6700
    elif wc and ad:
        price = 1000+9000
    elif pc and ad:
        price = 6700+9000
    elif wc:
        price=1000
    elif pc:
        price=6700
    else:
        price=9000

    if (coupon == "Yes"):
        cwrite=st.text_input("Use coupon code 'FREE' and hit enter ")
        if cwrite=='FREE':
            discount_price = price - 40
            st.success("CONGRATULATIONS! 40 UNITS HAVE BEEN DEDUCTED FROM YOUR CART! ")
            st.info(f"Final Price after Coupon: {discount_price}")
        elif cwrite=='':
            st.warning("Please write the code")
        else:
            st.error("Incorrect Coupon code.Try again.")
    else:
        discount_price=price

    def are_fields_filled(cname, age, mobile, email, address):
        return cname != "" and email != "" and age > 0 and mobile !="" and address !=""

    submit_button = st.button("SUBMIT THE CART")


    if submit_button:
        pattern = "^91"
        test = mobile
        result = re.match(pattern, test)
        if result:
            def shop_elec():
                sql = "insert into electronics(cname,age,mobile,email,address,price) values (%s,%s,%s,%s,%s,%s)"
                val = (cname, age, mobile, email, address,discount_price)
                try:
                    if are_fields_filled(cname, age, mobile, email, address):
                        # inserting the values into the table
                        cur.execute(sql, val)
                        myconn.commit()
                        # committed the transaction
                        st.success("Transaction has been saved")
                        st.info(f"{cur.rowcount} record inserted")
                        cur.execute("SELECT * FROM electronics WHERE cname = %s", (cname,))
                        records = cur.fetchall()
                        if records:
                            st.subheader("Recently Added Record:")
                            st.table(records)
                    else:
                        st.warning("Please fill in all the required fields properly.")
                except Exception as e:
                    st.error(f"cannot process {e}")
                    myconn.rollback()
            shop_elec()
        else:
            st.error("Enter mobile number properly")
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------


if option=="VEGETABLES":
    st.subheader("VEGETABLE ITEMS")
    col1, col2, col3 = st.columns(3)
    with col1:  # adding content to col1
        img1 = Image.open('tomato.jpg')  # loading the static
        st.image(img1,caption="Tomato")  # displaying static on webpage
        st.info("Price=210")
    with col2:  # adding content to col2
        img2 = Image.open('onion.jpg')  # loading the static
        st.image(img2,caption="Onions")  # displaying static on webpage
        st.info("Price=54")
    with col3:  # adding content to col3
        img3 = Image.open('potato.jpg')  # loading the static
        st.image(img3,caption="Potato")  # displaying static on webpage
        st.info("Price=80")

    s_title = st.text("Choose")
    tomato = st.checkbox('Tomato')
    onion=st.checkbox('Onion')
    potato=st.checkbox('Potato')
    cname = st.text_input("Enter customer name: ")
    age=st.slider("Enter your age:",min_value=1,max_value=100)
    mobile = st.text_input("Enter mobile number: ")
    email=st.text_input("Enter email: ")
    address=st.text_input("Enter address: ")
    coupon = st.selectbox("Would you like to use coupon and deduct 40 units", options=["No", "Yes"])


    if tomato and onion:
        price = 210+54
    elif tomato and potato:
        price = 210+80
    elif onion and potato:
        price = 54+80
    elif tomato:
        price=210
    elif onion:
        price=54
    else:
        price=80

    if (coupon == "Yes"):
        cwrite=st.text_input("Use coupon code 'FREE' and hit enter ")
        if cwrite=='FREE':
            discount_price = price - 40
            st.success("CONGRATULATIONS! 40 UNITS HAVE BEEN DEDUCTED FROM YOUR CART! ")
            st.info(f"Final Price after Coupon: {discount_price}")
        elif cwrite=='':
            st.warning("Please write the code")
        else:
            st.error("Incorrect Coupon code.Try again.")
    else:
        discount_price=price

    def are_fields_filled(cname, age, mobile, email, address):
        return cname != "" and email != "" and age > 0 and mobile !="" and address !=""

    submit_button = st.button("SUBMIT THE CART")


    if submit_button:
        pattern = "^91"
        test = mobile
        result = re.match(pattern, test)
        if result:
            def shop_vege():
                sql = "insert into vegetables(cname,age,mobile,email,address,price) values (%s,%s,%s,%s,%s,%s)"
                val = (cname, age, mobile, email, address,discount_price)
                try:
                    if are_fields_filled(cname, age, mobile, email, address):
                        # inserting the values into the table
                        cur.execute(sql, val)
                        myconn.commit()
                        # committed the transaction
                        st.success("Transaction has been saved")
                        st.info(f"{cur.rowcount} record inserted")
                        cur.execute("SELECT * FROM vegetables WHERE cname = %s", (cname,))
                        records = cur.fetchall()
                        if records:
                            st.subheader("Recently Added Record:")
                            st.table(records)
                    else:
                        st.warning("Please fill in all the required fields properly.")
                except Exception as e:
                    st.error(f"cannot process {e}")
                    myconn.rollback()
                st.info(f"{cur.rowcount} record inserted")
            shop_vege()
        else:
            st.error("Enter mobile number properly")


#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------


if option=="GROCERY":
    st.subheader("GROCERY ITEMS")
    col1, col2, col3 = st.columns(3)
    with col1:  # adding content to col1
        img1 = Image.open('cheese.jpg')  # loading the static
        st.image(img1,caption="Cheese")  # displaying static on webpage
        st.info("Price=190 ")
    with col2:  # adding content to col2
        img2 = Image.open('chips.png')  # loading the static
        st.image(img2,caption="Lays Chips")  # displaying static on webpage
        st.info("Price=69")
    with col3:  # adding content to col3
        img3 = Image.open('chocolate.jpg')  # loading the static
        st.image(img3,caption="Kit Kat Chocolate")  # displaying static on webpage
        st.info("Price=520")

    s_title = st.text("Choose")
    cheese = st.checkbox('Cheese')
    chips=st.checkbox('Lays Chips')
    chocolate=st.checkbox('Kit Kat Chocolate')

    cname = st.text_input("Enter customer name: ")
    age=st.slider("Enter your age:",min_value=1,max_value=100)
    mobile = st.text_input("Enter mobile number: ")
    email=st.text_input("Enter email: ")
    address=st.text_input("Enter address: ")
    coupon = st.selectbox("Would you like to use coupon and deduct 40 units", options=["No", "Yes"])

    if cheese and chips:
        price = 190+69
    elif cheese and chocolate:
        price = 190+520
    elif chips and chocolate:
        price = 69+520
    elif cheese:
        price=190
    elif chips:
        price=69
    else:
        price=520


    if (coupon == "Yes"):
        cwrite=st.text_input("Use coupon code 'FREE' and hit enter ")
        if cwrite=='FREE':
            discount_price = price - 40
            st.success("CONGRATULATIONS! 40 UNITS HAVE BEEN DEDUCTED FROM YOUR CART! ")
            st.info(f"Final Price after Coupon: {discount_price}")
        elif cwrite=='':
            st.warning("Please write the code")
        else:
            st.error("Incorrect Coupon code.Try again.")
    else:
        discount_price=price


    def are_fields_filled(cname, age, mobile, email, address):
        return cname != "" and email != "" and age > 0 and mobile != "" and address != ""

    submit_button = st.button("SUBMIT THE CART")


    if submit_button:
        pattern = "^91"
        test = mobile
        result = re.match(pattern, test)
        if result:
            def shop_groc():
                sql = "insert into grocery(cname,age,mobile,email,address,price) values (%s,%s,%s,%s,%s,%s)"
                val = (cname, age, mobile, email, address,discount_price)
                try:
                    if are_fields_filled(cname, age, mobile, email, address):
                        # inserting the values into the table
                        cur.execute(sql, val)
                        myconn.commit()
                        # committed the transaction
                        st.success("Transaction has been saved")
                        st.info(f"{cur.rowcount} record inserted")
                        cur.execute("SELECT * FROM grocery WHERE cname = %s", (cname,))
                        records = cur.fetchall()
                        if records:
                            st.subheader("Recently Added Record:")
                            st.table(records)
                    else:
                        st.warning("Please fill in all the required fields properly.")
                except Exception as e:
                    st.error(f"cannot process {e}")
                    myconn.rollback()
                st.info(f"{cur.rowcount} record inserted")
            shop_groc()
        else:
            st.error("Enter mobile number properly")
