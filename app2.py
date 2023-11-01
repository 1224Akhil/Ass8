try:
 
    from enum import Enum
    from io import BytesIO, StringIO
    from typing import Union
    import os
    import sys
    import pandas as pd
    import streamlit as st
    import pickle
except Exception as e:
    print(e)
 
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
 
 
class FileUpload(object):
    st.title(" Spam detection with stream Lit")
    def __init__(self):
        self.fileTypes = ["csv"]
 
    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        st.info(__doc__)
        st.markdown(STYLE, unsafe_allow_html=True)

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
        
        tfvect = pickle.load(open('file2.pkl','rb'))
        model = pickle.load(open('file1.pkl','rb'))
        
        if st.button("Predict"):
             textInput = [textInput]
             textInput = tfvect.transform(textInput)
             output  = model.predict(textInput)
             st.success('The given text is {} with {:0.2f} accuracy'.format(output,score))

             if output == "ham":
                 st.markdown(notSpam_html,unsafe_allow_html=True)
             else:
                 st.markdown(spam_html,unsafe_allow_html=True)
        
        file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " + ", ".join(["csv"]))
            return
        
        content = file.getvalue()
        data = pd.read_csv(file,encoding='ISO-8859-1')
        st.dataframe(data.head())

        message = data['v2']
        
 

        preds = []

        for i in message:
           i = [i]
           i = tfvect.transform(i)
           ans  = model.predict(i)
           preds.append(ans)

        st.dataframe(preds)

        data['ans'] = preds

        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
          "Press to Download",
          csv,
          "file.csv",
          "text/csv",
          key='download-csv'
        )

 
if __name__ ==  "__main__":
    helper = FileUpload()
    helper.run()
