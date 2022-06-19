import numpy as np
from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    ss = StandardScaler()
    final_features =  ss.fit_transform(final_features)
    
    prediction = model.predict(final_features)
    probability = model.predict_proba(final_features)
    print(probability)



    prob_benign = probability[0][0] 
    prob_malignant = probability[0][1]
    
    prediction =   "Benign Tumour"  if prediction[0] == 0 else "Malignant Tumour"

    print(prediction)

    return render_template('index.html', predict = prediction, prob_benign = prob_benign, prob_malignant = prob_malignant)

    
        



if __name__=='__main__':
    app.run(debug=True)