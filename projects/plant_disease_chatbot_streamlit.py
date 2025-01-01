import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import xgboost as xgb

# Function to handle user queries related to plant diseases
def query(user_input):
    response = []

    if "symptoms" in user_input.lower():
        response.append("Chatbot: Common symptoms include spots on leaves, wilting, yellowing, stunted growth, lesions, mold, or fungal growth.")
    elif "causes" in user_input.lower():
        response.append("Chatbot: Plant diseases can be caused by various factors such as fungi, bacteria, viruses, environmental stressors, pests, inadequate nutrition, or poor soil conditions.")
    elif "prevent" in user_input.lower():
        response.append("Chatbot: Ensure proper plant nutrition, watering, good soil drainage, spacing between plants, regular inspections, crop rotation, and use of disease-resistant plant varieties.")
    elif "treatment" in user_input.lower() or "remedy" in user_input.lower():
        response.append("Chatbot: Organic treatments may include neem oil, baking soda solutions, copper fungicides, or using natural predators to control pests. Soil health enhancement with compost or natural amendments can also help.")
    elif "treat" in user_input.lower():
        response.append("Chatbot: Organic treatments may include neem oil, baking soda solutions, copper fungicides, or using natural predators to control pests. Soil health enhancement with compost or natural amendments can also help.")
    if "watering" in user_input.lower() or "how much water" in user_input.lower():
        response.append(
            "Chatbot: Different plants have varying water needs. Generally, water when the top inch of soil is dry. Adjust watering based on season, plant type, and local climate conditions.")
    if "sunlight" in user_input.lower() or "light requirement" in user_input.lower():
        response.append(
            "Chatbot: Most plants need adequate sunlight for photosynthesis. Ensure they receive the recommended amount—full sun, partial shade, or full shade—based on their species.")
    if "pruning" in user_input.lower() or "trimming" in user_input.lower():
        response.append(
            "Chatbot: Regularly prune dead or diseased branches to encourage healthy growth. Use clean, sharp tools and prune during the plant's dormant season for best results.")
    if "fertilize" in user_input.lower() or "nutrition" in user_input.lower():
        response.append(
            "Chatbot: Fertilize plants during their growing season using appropriate fertilizers. Follow instructions carefully to avoid over-fertilization, which can harm plants.")
    if "pests" in user_input.lower() or "insects" in user_input.lower():
        response.append(
            "Chatbot: To control pests, consider natural predators, insecticidal soaps, or neem oil. Regularly inspect plants for signs of infestation and take preventive measures.")
    if "transplant" in user_input.lower() or "repotting" in user_input.lower():
        response.append(
            "Chatbot: When transplanting, choose a larger pot with good drainage. Gently loosen the roots before replanting and water thoroughly. Avoid disturbing the roots excessively.")
    if "winter care" in user_input.lower() or "cold weather" in user_input.lower():
        response.append(
            "Chatbot: In winter, protect sensitive plants from frost by covering them or bringing them indoors. Reduce watering and avoid fertilizing dormant plants.")
    if "companion planting" in user_input.lower() or "plant combinations" in user_input.lower():
        response.append(
            "Chatbot: Some plants benefit from being grown together. For instance, planting basil near tomatoes can repel pests. Research companion planting for specific plant pairs.")
    if "soil improvement" in user_input.lower() or "soil quality" in user_input.lower():
        response.append(
            "Chatbot: Enhance soil quality by adding compost, mulch, or organic matter. Conduct soil tests to understand its pH and nutrient levels for better plant growth.")
    if "overwatering" in user_input.lower() or "waterlogged soil" in user_input.lower():
        response.append(
            "Chatbot: Symptoms of overwatering include wilting, yellowing, and root rot. Allow the soil to dry out between waterings and ensure proper drainage to prevent waterlogged conditions.")
    if "pollinators" in user_input.lower() or "attract bees" in user_input.lower():
        response.append(
            "Chatbot: Attract bees and other pollinators by planting flowers such as lavender, sunflowers, and bee balm. Avoid using pesticides harmful to pollinators.")
    if "organic pest control" in user_input.lower() or "natural insect repellents" in user_input.lower():
        response.append(
            "Chatbot: Use natural repellents like garlic spray, diatomaceous earth, or companion planting with marigolds or mint to deter pests without harming beneficial insects.")
    if "soil pH" in user_input.lower() or "acidic soil" in user_input.lower() or "alkaline soil" in user_input.lower():
        response.append(
            "Chatbot: Test soil pH to understand acidity or alkalinity. Certain plants prefer specific pH levels; adjust soil acidity with additives like sulfur or lime accordingly.")
    if "beneficial microorganisms" in user_input.lower() or "soil microbes" in user_input.lower():
        response.append(
            "Chatbot: Beneficial soil microbes enhance nutrient availability. Use organic matter and compost to promote a healthy microbial environment in the soil.")
    else:
        response.append("Chatbot: I'm sorry, I don't have information on that. Please ask a different question.")

    return response

# Sidebar selection
st.sidebar.title('Navigation')
selection = st.sidebar.radio('Go to:', ('Plant Disease Chatbot', 'Plant Disease Detector'))

# Set up the RandomForest Classifier data
data = pd.read_csv(r"C:\Users\rajes\OneDrive\Desktop\TARP.csv")

air_mean = data['Air temperature (C)'].mean()
data['Air temperature (C)'].fillna(air_mean,inplace=True)# replacing missing values with mean

wind_mean = data['Wind speed (Km/h)'].mean()
data['Wind speed (Km/h)'].fillna(wind_mean,inplace=True)

hum_mean = data['Air humidity (%)'].mean()
data['Air humidity (%)'].fillna(hum_mean,inplace=True)

gus_mean = data['Wind gust (Km/h)'].mean()
data['Wind gust (Km/h)'].fillna(gus_mean,inplace=True)

pres_mean = data['Pressure (KPa)'].mean()
data['Pressure (KPa)'].fillna(pres_mean,inplace=True)

ph_mean = data['ph'].mean()
data['ph'].fillna(ph_mean,inplace=True)

rain_mean = data['rainfall'].mean()
data['rainfall'].fillna(rain_mean,inplace=True)

status_text_to_num = {'ON':1 , 'OFF':0}
data['Status'] = data['Status'].map(status_text_to_num)

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.fit_transform(x_test)

mdl = xgb.XGBClassifier()
mdl.fit(x_train_scaled,y_train)

y_pred = mdl.predict(x_test_scaled)
y_actual = y_test

if selection == 'Plant Disease Chatbot':
    # Initialize session state for chatbot
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Main chatbot loop
    st.title("Plant Disease Chatbot")
    user_input = st.text_input("Ask a general question about plant disease:")
    if st.button("Send"):
        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append(f"You: {user_input}")

            # Generate a response and add it to chat history
            response = query(user_input)
            st.session_state.chat_history.append(response[0])

    # Display the conversation history with custom styling
    for chat in st.session_state.chat_history:
        st.markdown(f'<p class="chat-text">{chat}</p>', unsafe_allow_html=True)

elif selection == 'Plant Disease Detector':
    st.title("Plant Disease Detector")

    # User inputs for prediction
    soil_moisture = st.number_input('Enter soil moisture:',step=1)
    temperature = st.number_input('Enter temperature:',step=1)
    soil_humidity = st.number_input('Enter soil humidity:',step=1)
    air_temperature = st.number_input('Enter air temperature:')
    wind_speed = st.number_input('Enter wind speed:')
    air_humidity = st.number_input('Enter air humidity:')
    wind_gust = st.number_input('Enter wind gust:')
    pressure = st.number_input('Enter air pressure:')
    ph = st.number_input('Enter pH:')
    rainfall = st.number_input('Enter rainfall:')

    new_input = [
        [soil_moisture, temperature, soil_humidity, air_temperature, wind_speed, air_humidity, wind_gust, pressure, ph,
         rainfall]]
    new_user_input_scaled = sc.transform(new_input)

    new_output = mdl.predict(new_user_input_scaled)

    if st.button("Predict"):
        if new_output[0] == 0:
            st.error('\n Patient is Unhealthy')
        elif new_output[0] == 1:
            st.success('\n Patient is Healthy')

        st.info(f'Accuracy: {accuracy_score(y_pred,y_actual)}')
        st.info(f'Precision: {precision_score(y_pred, y_actual)}')
        st.info(f'Recall Score: {recall_score(y_pred, y_actual)}')
        st.info(f'F1 score: {f1_score(y_pred, y_actual)}')
