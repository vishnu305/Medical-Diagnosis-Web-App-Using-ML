from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)
@app.route('/')
def front_page():
    return render_template('frontpage.html')
@app.route('/liver',methods=['GET','POST'])
def cancer_page():
    if request.method == 'GET':
        return render_template('cancer.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        total_bilirubin = request.form['chest']
        direct_bilirubin = request.form['trestbps']
        alkaline_phosphotase = request.form['chol']
        alamine_aminotransferase = request.form['fbs']
        aspartate_aminotransferase = request.form['restecg']
        total_proteins = request.form['thalach']
        albumin = request.form['exang']
        albumin_and_globulin_ratio = request.form['oldpeak']

        #liver_dataset = pd.read_csv('C:/Users/DIVVELA VISHNU/Desktop/Disease Detection Project/Heart Problem Detection/Flask Development/venv/indian_liver_patient.csv')
        # liver_dataset = pd.read_csv('indian_liver_patient.csv')
        # liver_dataset['Gender'] = liver_dataset['Gender'].map({'Male': 1, 'Female': 2})
        # liver_dataset.dropna(inplace=True)
        # X = liver_dataset.drop(columns='Dataset', axis=1)
        # Y = liver_dataset['Dataset']
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
        # model1 = RandomForestClassifier(n_estimators = 100)
        # model1.fit(X_train, Y_train)
        model1 = pickle.load(open('./static/liver_model.pkl','rb'))
        input_data = (age,sex,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio)
        print(input_data)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model1.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 2):
            senddata='According to the given details person does not have Liver Disease'
        else:
            senddata='According to the given details chances of having Liver Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)

@app.route('/heart',methods=['GET','POST'])
def heart_page():
    if request.method == 'GET':
        return render_template('heart.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        chest = request.form['chest']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        #heart_dataset = pd.read_csv('C:/Users/DIVVELA VISHNU/Desktop/Disease Detection Project/Heart Problem Detection/Flask Development/venv/heart.csv')
        # heart_dataset = pd.read_csv('heart.csv')
        # X = heart_dataset.drop(columns='target', axis=1)
        # Y = heart_dataset['target']
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
        # model1 = LogisticRegression(solver='lbfgs', max_iter=1000)
        # model1.fit(X_train.values, Y_train.values)
        model2 = pickle.load(open('./static/heart_model.pkl','rb'))
        input_data = [age,sex,chest,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        for i in range(len(input_data)):
            input_data[i]=float(input_data[i])
        print(input_data)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model2.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 0):
            senddata='According to the given details person does not have Heart Disease'
        else:
            senddata='According to the given details chances of having Heart Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata)
if __name__ == '__main__':
    app.run()