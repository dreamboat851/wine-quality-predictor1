import os
import streamlit as st
import joblib
import numpy as np
import pickle

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if value==val:
            return key
            
# Load Models
def load_model_n_predict(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def main():
    """ ML App with Streamlit"""
    st.title('Wine Quality Predictor')
    st.subheader('ML Prediction App with Streamlit')

    # RECEIVE USER INPUT

    fixed_acidity = st.slider("Select fixed acidity level",3.800,14.200)
    volatile_acidity = st.slider("Select volatile acidity level",0.08,1.1)
    citric_acid = st.slider("Select citric acid level",0.000,1.660)
    residual_sugar = st.slider("Select residual sugar level",0.500,66.000)
    chloride = st.slider("Select chloride level",0.009,0.346)
    free_sulfur_dioxide = st.slider("Select free sulfur dioxide level",2.000,289.000)
    total_sulfur_dioxide = st.slider("Select total sulfur dioxide level",9.0,440.0)
    density = st.number_input("Select density",0.980,1.040)
    pH = st.number_input("Select the pH level",2.700,3.900)
    sulfates = st.slider("Select the sulfate level",0.200,1.100)
    alcohol = st.slider("Select the alcohol level",8.000,14.200)

    # RESULT OF USER INPUT
    selected_options = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chloride, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulfates, alcohol]
    sample_data = np.array(selected_options).reshape(1, -1)
    st.info(selected_options)
    st.text("Using this encoding for prediction")
    st.success(selected_options)

    # MAKING PREDICTION
    ml_methods = ['decision tree','logistic regression','random forest','support vector machine']
    model_choice = st.selectbox('Model Choice',ml_methods)
    st.subheader("Prediction")
    prediction_label = {"Low quality":0, "High quality":1}
    if st.button("Predict"):
        if model_choice == 'decision tree':
            model_predictor = load_model_n_predict("wineQuality_v1_decision_tree.sav")
            prediction = model_predictor.predict(sample_data)
        elif model_choice == 'logistic regression':
            model_predictor = load_model_n_predict("wineQuality_v1_LR.sav")
            prediction = model_predictor.predict(sample_data)
        elif model_choice == 'random forest':
            model_predictor = load_model_n_predict("wineQuality_v1_random_forest.sav")
            prediction = model_predictor.predict(sample_data)
        elif model_choice == 'support vector machine':
            model_predictor = load_model_n_predict("wineQuality_v1_SVC.sav")
            prediction = model_predictor.predict(sample_data)
        print(prediction[0])
        print(type(prediction[0]))
        final_result = get_key(prediction[0],prediction_label)
        st.success("Predicted Quality is :: {}".format(final_result))

if __name__ == '__main__':
	main()


    

    

    

    

    

    
