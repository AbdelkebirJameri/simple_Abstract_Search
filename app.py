import streamlit as st
import numpy as np
import pickle
import pandas as pd
import nltk


from processing import get_embedding,out_dict,get_sim







def Ranked_documents(query):
    query_words = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(query.lower())],dtype=float), axis=0))
    rank = []
    for k,v in out_dict.items():
        rank.append((k, get_sim(query_words, v)))
        rank = sorted(rank,key=lambda t: t[1], reverse=True)
    print('Ranked Documents :')
    return rank[:3]


















def predicte():
    st.title("PFE SIMPLE SEARCH ENGINE ")
    
    
    input= st.text_input('SEARCH ', 'Enter your query')
    
    
    submit = st.button("search")
    if submit:
        

        predire=Ranked_documents(input)
        
        for i in predire:
            st.text("check this result")
            st.markdown(i,unsafe_allow_html=True)
            st.markdown("""<hr style="height:6px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
            
       
    




predicte()


