import joblib

def predict(form):
    # Cleans the used model and encoder in the past
    model = None
    encoder = None

    # Get input values from form
    sale = form.get('sales')
    features = [x for x in form.values()]

    usedModel = 'Random Forest' if (sale == 'NA' or sale == 'JP') else 'XGB'
    r2_score, deviation = getModelDetails(sale)

    if sale == None:
        modelPath = 'content/WithoutSales-Model.joblib'
        encoderPath = 'content/WithoutSales-Encoder.joblib'
        if len(features) == 8:
            features.pop(3)
    else:
        modelPath = 'content/' + sale + '-Model.joblib'
        encoderPath = 'content/' + sale + '-Encoder.joblib'
        features.pop(3)
    
    # Convert string to float for model 
    formInput = [float(item) for item in features]
    
    model = joblib.load(open(modelPath, 'rb'))
    encoder = joblib.load(open(encoderPath, 'rb'))

    result = model.predict(encoder.transform([formInput]))
    return result[0], usedModel, r2_score, deviation

def getModelDetails(sale = 'Without'):
    # returns R2_Score, Standard Deviation
    if sale == 'NA':
        return '0.91', '1.11'
    if sale == 'EU':
        return '0.84', '0.08'
    if sale == 'JP':
        return '0.63', '0.06'
    if sale == 'Other':
        return '0.74', '0.27'
    return '0.43', '1.09'     # Without Sales