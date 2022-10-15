import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

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
    liste_input = ["V34", "V94", "V70", "V91", "V294", "C8", "V308", "V281", "C12", "C1", "C14", "D2"]
    read_dictionary = np.load(r"C:\Users\hp\Desktop\Projet Encadré\Deployment avec Flask\my_dict_val_deflt.npy",allow_pickle='TRUE').item()
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    k = 0
    for ky in list(read_dictionary.keys()):
        if (ky in liste_input) and (len(list(read_dictionary.keys())) == len(final_features)):
            read_dictionary[ky] = final_features[k]
            k = k + 1
    
    interf = []
    final_features = pd.DataFrame()
    k = 0
    for ky in list(read_dictionary.keys()):
        interf.append(list(read_dictionary.values())[k])
        final_features[ky] = list(read_dictionary.values())[k]
        k = k + 1
        

    final_features.loc[0] = interf

    prediction = model.predict(final_features)

    output = int(prediction[0])
    if output == 0:
        return render_template('index.html', prediction_text='La transaction est Bonne !!!  {}'.format("(^_^)"))
    return render_template('index.html', prediction_text='La transaction est Frauduleuse !!!  {}'.format("(=_=)"))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

## liste_input = ["V34", "V94", "V70", "V91", "V294", "C8", "V308", "V281", "C12", "C1", "C14", "addr2"]

if __name__ == "__main__":
    app.run(debug=True)