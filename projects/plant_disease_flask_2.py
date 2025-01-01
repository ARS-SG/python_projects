from flask import Flask, render_template, request

app = Flask(__name__)

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
    else:
        response.append("Chatbot: I'm sorry, I don't have information on that. Please ask a different question.")

    return "<br>".join(response)

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/result', methods=['POST'])
def result():
    user_input = request.form['user_input']
    return query(user_input)

@app.route('/sidebar')
def sidebar():
    return render_template('sidebar.html')

if __name__ == '__main__':
    app.run(debug=True)
