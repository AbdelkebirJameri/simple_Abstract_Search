import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
import itertools
from nltk.tokenize import sent_tokenize, word_tokenize
import scipy
from scipy import spatial
from nltk.tokenize.toktok import ToktokTokenizer
import re
nltk.download('stopwords')
nltk.download('stopwords')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english') 




doc1=["""<b style="color:#46FA1A"> Title:  </b> development of a system for monitoring and predicting electricity consumption with Machine Learning and Deep Learning methods. <br>
<b style="color:#46FA1A"> Abstract:</b> Electrical energy is a very important factor for the evolution of Morocco, hence the interest of monitoring systems for controlling electricity consumption. This concept remains new in Moroccan companies, this framework is a part of our project of graduation studies, done within the company "Syntechnology", with the aim of designing and implementing a system of supervision which is based on monitoring and forecasting of electricity consumption. <br>
<b style="color:#46FA1A"> Student:</b> Asmae El Aouyed   <br>  <b style="color:#46FA1A"> Promotion:</b> 2020"""]

doc2=["""<b style="color:#46FA1A"> Title:</b> Development and improvement of the user interface in natural language: Developed solution for the conversion of natural language into query language. <br> <b style="color:#46FA1A"> Abstract:</b> This document is the result of a study project work that took place at Oracle Labs on the conversion of natural language queries to PGQL queries. The objective of this project is to improve the interface by solving the various sources of malfunctions and errors as well as the implementation of new functionalities.  <br>  <b style="color:#46FA1A"> Student: Hamza Fawzi.  <br>  promotion:2020 """]

doc3=["""<b style="color:#46FA1A">Title:</b> Robustness Enhancement in Chatbot NLU Models via Adversarial attacks, Data Augmentation and Adversarial Training.  <br> <b style="color:#46FA1A"> Abstract:</b> Nowadays, with the ongoing digital era, the revolution of intelligent machine and the advancement of artificial intelligent, machine learning and deep learning, machines have started to impersonate as human. Conversational software agents stimulated by natural language processing is known as chatbot, are an excellent example of such machine. <br><b style="color:#46FA1A"> Studen:</b> Ayoub AKENNAF  <br> <b style="color:#46FA1A"> Promotion:2021"""]

doc4=["""<b style="color:#46FA1A">Title: </b> Construction and analysis of a biomedical knowledge graph using Deep learning techniques <br>  <b style="color:#46FA1A"> Abstract:</b> Data science is a multidisciplinary field that uses knowledge from the fields of mathematics, statistics and technology to study and evaluate data. It is a subject that deals with the identification, representation and extraction of meaningful information from data sources in order to use them for commercial or other purposes. In this context, the objective of this thesis is to propose a generic approach for the construction of a knowledge graph using Deep Learning techniques, through textual data in the format of clinical notes  of people in order to analyze them. <br> <b style="color:#46FA1A"> Student: </b> HARNOUNE AYOUB  <br><b style="color:#46FA1A"> Promotion: 2021"""]

doc5=["""<b style="color:#46FA1A">Title: </b> Development and Deployment of a Web-Based SaaS (Software-as-a-Service) App for Hotels Internal Management  <br>  <b style="color:#46FA1A"> Abstract: </b> This report summarizes the work done as part of the graduation project within DigiMental. The project aim is to develop and deploy a web-based SaaS (Software-as-a-Service) app for hotels internal management. Including deployment automation and DevOps best practices. In 2019, the Morocco Tourism Observatory announced that nearly 7.544 million tourists had visited Morocco in late July 2019 (8% increase compared to the last year). Furthermore, experts forecast post-COVID-19 tourism recovery. To meet the demand, hotels find it challenging to manage critical daily tasks and coordinate between staff members. Any delays or issues will lead to customer frustration, which will lead to reputation damage on booking platforms. <br><b style="color:#46FA1A">  Student: </b>  KERBAL Ilyas  <br> <b style="color:#46FA1A"> Promotion: 2020  """]

doc6=["""<b style="color:#46FA1A">Title:</b> Graph Neural Networks for better mapping of sanitation networks <br>  <b style="color:#46FA1A"> Abstract: </b> This report is the fruit of our work carried out as part of our end-of-study project at BergerLevrault. The objective of this project is to improve the maps of sewerage networks whichsuffer from a set of imperfections, namely imprecision, incompleteness and uncertainty on bothparts, the spatial part which represents the digital graph of these networks and which is madeup of two main objects: manholes and conduits, and the attribute part which describes thecharacteristics of these objects.Taking into account the graphical structure of these networks, this project aims at developingtwo algorithms that exploit this structure in order to derive the best characteristics of thesemaps. This aims to ensure, on the one hand, the reconstruction of sewer network maps throughthe prediction of the existence of a conduit between two manholes and, on the other hand, theprediction of attributes in order to complete the attribute table of these conduits. <br> <b style="color:#46FA1A">  Student: </b>  Reda Abdou <br> <b style="color:#46FA1A"> Promotion: 2020"""]


doc7=["""<b style="color:#46FA1A">Title: </b> Development and Deployment of the HR modules of an ERP for a companys internal management <br> <b style="color:#46FA1A">  Abstract: </b> This report summarizes the work done as part of my end-of-studies internship within Avaliance. The project aim is to develop and deploy a web-based ERP (Enterprise Resource Planning) app, with plans to adapt it to mobile phones in the future, for the company’s internal management. Including deployment automation and DevOps best practices. <br> <b style="color:#46FA1A"> Student: </b> Oussama BAOUI <br> <b style="color:#46FA1A"> Promotion: 2020"""]

doc8=["""<b style="color:#46FA1A">Title:</b>  Leveraging Natural Language Processing Techniques to Analyze Agricultural Text Data: A Case Study on Risk Assessment Approaches <br> <b style="color:#46FA1A"> Abstract:</b>  Detecting opportunities and threats from massive text data is a challenging task for most. Traditionally, companies would rely mainly on structured data to detect and predict risks, losing a huge amount of information that could be extracted from unstructured text data. Fortunately, industry 4.0 Technologies came to remedy this issue by innovating in data extraction and processing techniques, allowing us to understand and make use of Natural Language data and turning it into structures that a machine can process <br> <b style="color:#46FA1A"> Student: </b> Zineb Belaabid  <br> <b style="color:#46FA1A">  Promotion: 2019  """] 

doc9=["""<b style="color:#46FA1A">Title:</b>  In silico prediction of influenza virus (H9N2) antigenicity from its hemagglutinin sequence and determination of antigenic sites likely to induce vaccination escape post mutation. <br> <b style="color:#46FA1A">  Abstract:</b>  The present document is the outcome of my end-of-study project that took place at the Molecular Biology Laboratory of the Avian Pathology Unit of the Agronomic and Veterinary institute Hassan II. It aims to predict the antigenicity of influenza virus (H9N2) using genetic data i.e. its hemagglutinin sequence, as well as to determine positions on the hemagglutinin sequence that induce vaccination escape as a result of their mutation. <br>
 <b style="color:#46FA1A"> Student:</b> Sihame Fellah  <br> <b style="color:#46FA1A"> Promotion: 2020 """]



fin = doc1+doc2+doc3+doc4+doc5+doc6+doc7+doc8+doc9


def remove_stopwords(text, is_lower_case=False):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern," ",''.join(text))
    tokens = tokenizer.tokenize(text)
    tokens = [tok.strip() for tok in tokens]
    if is_lower_case:
        cleaned_token = [tok for tok in tokens if tok not in stopword_list]
    else:
        cleaned_tokens = [tok for tok in tokens if tok.lower() not in stopword_list]
    filtered_text = ' '.join(cleaned_tokens)
    return filtered_text


remove_stopwords(fin[0])



glove_vectors = {}
file = open('data1.txt', encoding = 'utf-8')
for line in file:
    values = line.split()
    word = values[0]
    vectors = np.asarray(values[1:])
    glove_vectors[word] = vectors
file.close()






vec_dimension = 300
def get_embedding(x):
    arr  = np.zeros(vec_dimension)
    text = str(x).split()
    for t in text:
        try:
            vec = glove_vectors.get(t).astype(float)
            arr = arr + vec
        except:
            pass
    arr = arr.reshape(1,-1)[0]
    return(arr/len(text))


# Getting average vector for each document
out_dict = {}
for sen in fin:
    average_vector = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(remove_stopwords(sen))]), axis=0))
    dict = { sen : (average_vector) }
    out_dict.update(dict)



out_dict    



def get_sim(query_embedding, average_vector_doc):
    sim = [(1 - scipy.spatial.distance.cosine(query_embedding, 
    average_vector_doc))]
    return sim



