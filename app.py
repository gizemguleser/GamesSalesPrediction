from flask import Flask, render_template, request
import predictGamesSales

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', prediction_text='')

@app.route('/predict',methods=['GET'])
def predict():
    globalSales, model, r2_score, deviation = predictGamesSales.predict(request.args)
    globalSales_text = 'Predicted Sales in Millions: {:.4f}'.format(globalSales)
    model_text = 'Model: ' + model
    score_text = 'R2 Score: ' + r2_score
    deviation_text = 'Standard Deviation: ' + deviation
    return globalSales_text + ',' + model_text + ',' + score_text + ',' + deviation_text

@app.route('/details')
def details():
    return render_template('details.html')

if __name__ == "__main__":
    app.run(debug=True)