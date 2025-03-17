import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('b_training_model.sav', 'rb'))
 
def banana_quality_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "Bad Quality Banana"
    else:
        return "Good Quality Banana"

def main():
    st.title("Banana Quality Predictor 🍌")
    st.write("This is a simple Banana Quality Predictor web app. Please enter the following details to get the prediction.")
    st.write("")

    Size = st.text_input('Size')
    Weight = st.text_input('Weight')
    Sweetness = st.text_input('Sweetness')
    Softness = st.text_input('Softness')
    HarvestTime = st.text_input('Harvest Time')
    Ripeness = st.text_input('Ripeness')
    Acidity = st.text_input('Acidity')

    pred_diag = ''

    if st.button('Predict'):
        pred_diag = banana_quality_prediction([Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity])

    st.success(pred_diag)

if __name__ == '__main__':
    main()
