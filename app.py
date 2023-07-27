import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from io import StringIO
  
# loading in the model to predict on the data
pickle_in = open('cc200_pca_svm_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def main():
    st.title("Autism Diagnosis Model")
    st.write("Using Machine Learning Algorithms")
    
    
    

    uploaded_file = st.file_uploader("Choose a file", type="npy")
    image_reshape = np.load(uploaded_file)
    print(type(image_reshape))
    #image_reshape = np.nan_to_num(image_reshape)
    image_reshape = np.array(image_reshape).reshape(1,-1)
    #image_reshape_fill = np.array(image_reshape).filln(0)
    image_reshape[np.isnan(image_reshape)] = 0
#     b = np.reshape(image_reshape, (1, -1))
  
          
    # Display the prediction result
    prediction = classifier.predict(image_reshape)
    st.subheader("Prediction Result:")
    if prediction == 0:
        st.write("The model predicts the individual is not diagnosed with autism.")
    else:
        st.write("The model predicts the individual is diagnosed with autism.")
     
    st.caption("Created by: \n Olatunji Oyindamola Boluwaitfe - CSC/2016/089")




if __name__ == "__main__":
    main()
          
          
          
#     if uploaded_file is not None:
#         # To read file as bytes:
#         bytes_data = uploaded_file.getvalue()
#         st.write(bytes_data)
    
#         # To convert to a string based IO:
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         st.write(stringio)

#         # To read file as string:
#         string_data = stringio.read()
#         st.write(string_data)
        
         # Sample Data
    
#   st.write("Upload data:\n")
#   input_img  = st.file_uploader("Upload Image", type ='npy')
#   input_msk = st.file_uploader("Upload Mask", type ='npy')

    # Can be used wherever a "file-like" object is accepted:
#     dataframe = pd.read_csv(uploaded_file)
#     st.write(dataframe)


   # Make predictions using the trained SVM model
#     scaled_data = uploaded_file.reshape(1, -1) 
#     #prediction = classifier.predict(np.reshape(image_reshape, (-1,1)))
    #prediction = classifier.predict(image_reshape.reshape(-1,1))

    # Display the prediction result
#    st.subheader("Prediction Result:")
#     if prediction == 0:
#         st.write("The model predicts the individual is not diagnosed with autism.")
#     else:
#         st.write("The model predicts the individual is diagnosed with autism.")

#     st.subheader("Model Accuracy:")
#     st.write(f"Accuracy: {accuracy:.2f}")
#     st.subheader("Model Performance:")
#     st.write("SVM:")
#     st.write(f"Accuracy: {svm_accuracy:.2f}")
#     st.write(f"Sensitivity: {svm_sensitivity:.2f}")
#     st.write(f"Specificity: {svm_specificity:.2f}")



# if __name__ == "__main__":
#     main()
