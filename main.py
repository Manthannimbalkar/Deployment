from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('diabetes_79.pkl')

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/blogs')
def gallary():
    return render_template('blog.html')

@app.route('/form', methods=['post'])
def form():
    preg = request.form.get("preg")
    pres = request.form.get("pres")
    plas = request.form.get("plas")
    skin = request.form.get("skin")
    test = request.form.get("test")
    mass = request.form.get("mass")
    predi = request.form.get("predi")
    age = request.form.get("age")

    result = model.predict([[int(preg),int(pres),int(plas),int(skin),int(test),int(mass), int(predi), int(age)]])

    if result[0]==1:
        data = 'Person is Diabetic'
    else:
        data = 'Person is not Diabetic'
    return data

app.run(host='0.0.0.0',port=8020)