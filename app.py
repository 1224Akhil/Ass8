import streamlit as st
import pickle
def predict(text):
  tfvect = pickle.load(open('file2.pkl','rb'))
  model = pickle.load(open('file1.pkl','rb'))
  text = [text]
  text = tfvect.transform(text)
  preds = model.predict(text)
  if preds != 'ham':
            return " Spam "
  else : 
      return "Not Spam"

def main():
    st.title(" Spam detection with stream Lit")
    try:
        import os
        import sys
        import pandas as pd
        from io import BytesIO,StringIO
        score = pickle.load(open('file3.pkl','rb'))
        html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">Spam Prediction ML App </h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        textInput = st.text_input("text","Type Here")
        notSpam_html="""  
        <div style="background-color:#F4D03F;padding:10px >
        <h2 style="color:white;text-align:center;">Text is not Spam</h2>
        </div>
        """
        spam_html="""  
        <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black ;text-align:center;">Text is Spam </h2>
        </div>
        """

        if st.button("Predict"):
            output= predict(textInput)
            st.success('The given text is {} with {:0.2f} accuracy'.format(output,score))

            if output == "Not Spam":
                st.markdown(notSpam_html,unsafe_allow_html=True)
            else:
                st.markdown(spam_html,unsafe_allow_html=True)

if __name__=='__main__':
  main()