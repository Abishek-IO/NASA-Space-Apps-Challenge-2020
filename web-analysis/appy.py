import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output < 1:
        return render_template('index.html', prediction_text='Global Rise in Temperature will be {} °C. We seem to be on track, let us keep it up!'.format(output))
    elif output < 1.5:
        return render_template('index.html', prediction_text='Global Rise in Temperature will be {} °C. Let us be vigilant, things could get out of hand soon'.format(output))
    elif output < 2:
        return render_template('index.html', prediction_text='Global Rise in Temperature will be {} °C. We need to take extreme measures, the situation is extremely dangerous!'.format(output))
    elif output >= 2:
        return render_template('index.html', prediction_text='Global Rise in Temperature will be {} °C. Catastrophic situation!'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
