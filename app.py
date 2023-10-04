# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, request,render_template
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
log_reg_model = pkl.load(open('logisticreg.pkl','rb'))
dec_tree_model = pkl.load(open('decisiontree.pkl','rb'))
rndm_frst_model = pkl.load(open('randomforest.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/logreg')
def logreg_page():
    return render_template('lr_upload.html')

@app.route('/lrpredict',methods=['POST'])
def predict_logreg(): 
    f = request.files.get('fileupload')
    df = pd.read_excel(f)
    df_new = df.drop(columns=["Product ID"])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_new)
    df_scaled = pd.DataFrame(scaled_features,columns = df.columns[1:])
    df_selected =df_scaled.loc[:,"Air temperature [K]":"Tool wear [min]"]
    numpy_f = df_selected.to_numpy()
    y_preds = []
    output = []
    
    product_id = list(df["Product ID"])

    for i in range(len(numpy_f)):
        features = [numpy_f[i]]
        
        y_pr =log_reg_model.predict(features)
        y_p = y_pr.item()
        y_preds.append(y_p)
        
    for i in range(len(y_preds)):
        if(y_preds[i]==0):
            output.append("No failure will occur")
        else:
            output.append("Failure may occur")
    
        
    return render_template('check_p.html', Machine=product_id,Output=output)

@app.route('/dectr')
def decisiontree_page():
    return render_template('dt_upload.html')

@app.route('/dtpredict',methods=['POST'])
def predict_decisiontree(): 
    f = request.files.get('fileupload')
    df = pd.read_excel(f)
    df_new = df.drop(columns=["Product ID"])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_new)
    df_scaled = pd.DataFrame(scaled_features,columns = df.columns[1:])
    df_selected =df_scaled.loc[:,"Air temperature [K]":"Tool wear [min]"]
    numpy_f = df_selected.to_numpy()
    y_preds = []
    output = []
    
    product_id = list(df["Product ID"])

    for i in range(len(numpy_f)):
        features = [numpy_f[i]]
        
        y_pr = dec_tree_model.predict(features)
        y_p = y_pr.item()
        y_preds.append(y_p)
        
    for i in range(len(y_preds)):
        if(y_preds[i]==0):
            output.append("No failure will occur")
        else:
            output.append("Failure may occur")
    
        
    return render_template('check_p.html', Machine=product_id,Output=output)


@app.route('/rdmfrst')
def randomfrst_page():
    return render_template('rdmfrst_upload.html')

@app.route('/rfpredict',methods=['POST'])
def predict_randomfrst(): 
    f = request.files.get('fileupload')
    df = pd.read_excel(f)
    df_new = df.drop(columns=["Product ID"])
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_new)
    df_scaled = pd.DataFrame(scaled_features,columns = df.columns[1:])
    df_selected =df_scaled.loc[:,"Air temperature [K]":"Tool wear [min]"]
    numpy_f = df_selected.to_numpy()
    y_preds = []
    output = []
    
    product_id = list(df["Product ID"])

    for i in range(len(numpy_f)):
        features = [numpy_f[i]]
        
        y_pr = rndm_frst_model.predict(features)
        y_p = y_pr.item()
        y_preds.append(y_p)
        
    for i in range(len(y_preds)):
        if(y_preds[i]==0):
            output.append("No failure will occur")
        else:
            output.append("Failure may occur")
    
        
    return render_template('check_p.html', Machine=product_id,Output=output)


#@app.route('/upload')
#def upload_route_summary():
#    return render_template('upload.html')

#@app.route('/file',methods=['POST'])
#def file_upload():
#       f = request.files.get('fileupload')
#       df = pd.read_csv(f, encoding='latin-1')
#       output = df.to_numpy()
#       output_arr = np.array(output)
       
#       return render_template('display.html', dict_output=output_arr)
    
if __name__ == "__main__":
    app.run(debug=True)    
        
    
    
    