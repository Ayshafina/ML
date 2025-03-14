import streamlit as st
import joblib

model=joblib.load("model.pkl")
l1=joblib.load("label.pkl")
sd=joblib.load("sd.pkl")

st.title("LUNG_CANCER")
st.write("Enter Data Description")

GENDER= st.selectbox('Enter the GENDER',['F','M'])
AGE=st.number_input("Enter your AGE")
SMOKING= st.number_input('Enter the SMOKING')
YELLOW_FINGERS= st.number_input("Enter the YELLOW_FINGERS")
ANXIETY = st.number_input("Enter the ANXIETY ")
PEER_PRESSURE=st.number_input('enter the PEER_PRESSURE')
CHRONIC_DISEASE	=st.number_input("Enter the CHRONIC DISEASE	")
FATIGUE=st.number_input("Enter the FATIGUE")
ALLERGY	=st.number_input("Enter the ALLERGY	")
WHEEZING=st.number_input("Enter the WHEEZING")
ALCOHOL_CONSUMING=st.number_input("Enter the ALCOHOL CONSUMING")
COUGHING=st.number_input("Enter the COUGHING")
SHORTNESS_OF_BREATH	=st.number_input("Enter the SHORTNESS OF BREATH")
SWALLOWING_DIFFICULTY=st.number_input("Enter the SWALLOWING DIFFICULTY")
CHEST_PAIN=st.number_input("Enter the CHEST PAIN")


GENDER=l1.transform([GENDER])[0]

if st.button("predict"):
    result=model.predict(sd.transform([[GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSUMING,
                                        COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]]))[0]
    st.success("the output is{}".format(result))
                                        
                                        
    


    


