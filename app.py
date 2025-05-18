from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models & encoder
with open('models/reg_model.pkl','rb') as f:
    reg = pickle.load(f)
with open('models/clf_model.pkl','rb') as f:
    clf = pickle.load(f)
with open('models/label_encoder.pkl','rb') as f:
    scaler, le_kelas = pickle.load(f)

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':

        LT = float(request.form['LT'])
        LB = float(request.form['LB'])
        JKT = float(request.form['JKT'])
        JKM = float(request.form['JKM'])
        GRS = 1 if request.form['GRS']=='ada' else 0
        
        # Preprocess input
        arr = np.array([[LT,LB,JKT,JKM,GRS]])
        arr_scaled = scaler.transform(arr)

        # Predict
        harga_pred = reg.predict(arr_scaled)[0]
        kelas_pred = le_kelas.inverse_transform(clf.predict(arr_scaled))[0]

        result = {'harga': round(harga_pred,2), 'kelas': kelas_pred}

    return render_template('index.html', result=result)

if __name__=='__main__':
    app.run(debug=True)