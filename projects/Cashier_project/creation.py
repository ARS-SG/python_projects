import mysql.connector

#create the connection object
myconn=mysql.connector.connect(host="localhost",user="root",passwd="",database="shopping")

#creating the cursor object
cur=myconn.cursor()

try:
    data=cur.execute("create table grocery(cname varchar(20) not null,age int(3) not null,mobile varchar(20) not null,email varchar(20) not null,address varchar(20) not null,price int(20) not null)")
    print("your table is created successfully")
except Exception as e:
    #myconn.rollback()
    print("cannot process",e)

myconn.close()
