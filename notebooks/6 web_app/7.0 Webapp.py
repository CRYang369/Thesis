import streamlit as st
import pandas as pd 
import numpy as np 
import joblib 
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt 

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
st.title("Cardiotocography results classification")
col1, col2=st.columns([0.4,  0.6])
with col1:

    LB = st.number_input('LB',  -500.0, 648.0, value=0.0)
    LB_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of baseline per 100 seconds</p>'
    st.markdown(LB_describe, unsafe_allow_html=True)

    AC = st.number_input('AC',  -500.0, 648.0, value=0.0) 
    AC_describe = '<p style="font-family:Courier; color:Black; font-size: 12px;">Number of accelerations per 100 seconds</p>'
    st.markdown(AC_describe, unsafe_allow_html=True)
    
    FM = st.number_input('FM', -500.0, 648.0, value=0.0)
    FM_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of fetal movements per 100 seconds</p>'
    st.markdown(FM_describe, unsafe_allow_html=True)

    UC = st.number_input('UC', -500.0, 648.0, value=0.0)
    UC_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of uterine contractions per 100 seconds</p>'
    st.markdown(UC_describe, unsafe_allow_html=True)

    DL = st.number_input('DL', -500.0, 648.0, value=0.0)
    DL_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of light decelerations per 100 seconds</p>'
    st.markdown(DL_describe, unsafe_allow_html=True)

    DP = st.number_input('DP', -500.0, 648.0, value=0.0)
    DP_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of prolongued deceleations per 100 seconds</p>'
    st.markdown(DP_describe, unsafe_allow_html=True)

    ASTV = st.number_input('ASTV', -500.0, 648.0, value=0.0)
    ASTV_describe = '<p style="font-family:Courier;  font-size: 12px;">Percentage of time with abnormal short term variability</p>'
    st.markdown(ASTV_describe, unsafe_allow_html=True)


    MSTV = st.number_input('MSTV', -500.0, 648.0, value=0.0)
    MSTV_describe = '<p style="font-family:Courier;  font-size: 12px;">Mean value of short term variablity</p>'
    st.markdown(MSTV_describe, unsafe_allow_html=True)


    ALTV = st.number_input('ALTV', -500.0, 648.0, value=0.0)    
    ALTV_describe = '<p style="font-family:Courier;  font-size: 12px;">Percentage of time with abnormal long term variability</p>'
    st.markdown(ALTV_describe, unsafe_allow_html=True)

    MLTV = st.number_input('MLTV', -500.0, 648.0, value=0.0)
    MLTV_describe = '<p style="font-family:Courier;  font-size: 12px;">Mean value of  long term variability</p>'
    st.markdown(MLTV_describe, unsafe_allow_html=True)


    Width = st.number_input('Width', -500.0, 648.0, value=0.0)
    Width_describe = '<p style="font-family:Courier;  font-size: 12px;">Width of FHR histogram</p>'
    st.markdown(Width_describe, unsafe_allow_html=True)


    Min = st.number_input('Min', -500.0, 648.0, value=0.0)
    Min_describe = '<p style="font-family:Courier;  font-size: 12px;">Minimum of FHR histogram</p>'
    st.markdown(Min_describe, unsafe_allow_html=True)


    Max = st.number_input('Max', -500.0, 648.0, value=0.0)
    Max_describe = '<p style="font-family:Courier;  font-size: 12px;">Maximun of FHR histogram</p>'
    st.markdown(Max_describe, unsafe_allow_html=True)

    Nmax = st.number_input('Nmax', -500.0, 648.0, value=0.0)
    Nmax_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of histogram peaks</p>'
    st.markdown(Nmax_describe, unsafe_allow_html=True)


    Nzeros = st.number_input('Nzeros', -500.0, 648.0, value=0.0)
    Nzeros_describe = '<p style="font-family:Courier;  font-size: 12px;">Number of histogram zeros</p>'
    st.markdown(Nzeros_describe, unsafe_allow_html=True)


    Mode = st.number_input('Mode', -500.0, 648.0, value=0.0)
    Mode_describe = '<p style="font-family:Courier;  font-size: 12px;">Histogram mode</p>'
    st.markdown(Mode_describe, unsafe_allow_html=True)


    Mean = st.number_input('Mean', -500.0, 648.0, value=0.0)
    Mean_describe = '<p style="font-family:Courier;  font-size: 12px;">Histogram mean</p>'
    st.markdown(Mean_describe, unsafe_allow_html=True)


    Median = st.number_input('Median', -500.0, 648.0, value=0.0)
    Median_describe = '<p style="font-family:Courier;  font-size: 12px;">Histogram median</p>'
    st.markdown(Median_describe, unsafe_allow_html=True)


    Variance = st.number_input('Variance', -500.0, 648.0, value=0.0)
    Variance_describe = '<p style="font-family:Courier;  font-size: 12px;">Histogram variance</p>'
    st.markdown(Variance_describe, unsafe_allow_html=True)


    Tendency = st.number_input('Tendency', -500.0, 648.0, value=0.0)
    Tendency_describe = '<p style="font-family:Courier;  font-size: 12px;">Histogram tendency</p>'
    st.markdown(Tendency_describe, unsafe_allow_html=True)



    
    if st.button("Get predictions"):

        with col2: 
             feature_list=[ LB, 
                        AC, FM, UC, DL
                        # , DS
                        , DP, ASTV, MSTV, ALTV, MLTV, Width, Min, Max, Nmax
                        , Nzeros
                        , Mode, Mean
                        , Median, Variance
                        , Tendency]

             pretty_result = {
                        'LB':LB,
                        'AC':AC
                        , 'FM':FM
                        , 'UC':UC
                        , 'DL':DL
                        , 'DP':DP
                        , 'ASTV':ASTV
                        , 'MSTV':MSTV
                        , 'ALTV':ALTV
                        , 'MLTV':MLTV
                        , 'Width':Width
                        , 'Min':Min
                        , 'Max':Max
                        , 'Nmax':Nmax
                        , 'Nzeros':Nzeros
                        , 'Mode':Mode 
                        , 'Mean':Mean
                        , 'Median':Median
                        , 'Variance':Variance
                        , 'Tendency':Tendency
             } 
             
     
             single_sample=np.array(feature_list).reshape(1,-1)
             loaded_model=joblib.load('rf_ctgnsp.model')
             prediction = loaded_model.predict(single_sample)
             pred_prob = loaded_model.predict_proba(single_sample)


             pred_probability_score = {"Normal":pred_prob[0][0]*100,"Suspect":pred_prob[0][1]*100,"Pathologic":pred_prob[0][2]*100}

            
     
             ax=list(pred_probability_score.keys())
             ay=list(pred_probability_score.values())

  
             fontdict = {'color': 'black',
                        'weight': 'normal',
                        'size': 3}

             st.header("Prediction")  
             plt.figure(figsize=(4,1))
        
             plt.barh(ax,ay,height = 0.3,color=['g','b','r',])
             plt.xticks(size=3)
             plt.yticks(size=3)
             plt.xlabel("Predicted probability",fontdict=fontdict)
  
             st.pyplot()

             st.header("Prediction Explanation")  
             df = pd.read_csv("CTGNSP selected 20 features.csv")
             x = df[[
                'LB', 
                'AC', 'FM', 'UC', 'DL'               
                , 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax'
                , 'Nzeros'
                , 'Mode', 'Mean',
                'Median', 'Variance'
                , 'Tendency'
                ]]
             feature_names = [
                'LB',
                'AC', 'FM', 'UC', 'DL'        
                , 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax'
                , 'Nzeros'
                , 'Mode', 'Mean',
                'Median', 'Variance'
                , 'Tendency'
                ]
             class_names = ['1','2','3']
             explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True,random_state=24)
   
             exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=20, top_labels=3)
             exp.show_in_notebook(show_table=True, show_all=False)



             new_exp = exp.as_list(label=0)
             label_limits = [i[0] for i in new_exp]
    
             label_scores = [i[1] for i in new_exp]

             plt.figure(figsize=(5,2.5))

             plt.barh(label_limits,label_scores,color=['g'])
             plt.xticks(size=4)
             plt.yticks(size=4)
             plt.xlabel("Impact on prediction ",fontdict=fontdict)
             plt.title('Normal',fontdict=fontdict)
             plt.legend(["Normal"],loc='best')
             

             st.pyplot()

             new_exp = exp.as_list(label=1)
             label_limits = [i[0] for i in new_exp]
     
             label_scores = [i[1] for i in new_exp]

             plt.figure(figsize=(5,2.5))

             plt.barh(label_limits,label_scores,color=['b'])
             plt.xticks(size=4)
             plt.yticks(size=4)
             plt.xlabel("Impact on prediction ",fontdict=fontdict)
             plt.title('Suspect',fontdict=fontdict)
             plt.legend(["Suspect"],loc='best')
             st.pyplot()



             new_exp = exp.as_list(label=2)
             label_limits = [i[0] for i in new_exp]
         
             label_scores = [i[1] for i in new_exp]

             plt.figure(figsize=(5,2.5))

             plt.barh(label_limits,label_scores,color=['r'])
             plt.xticks(size=4)
             plt.yticks(size=4)
             plt.xlabel("Impact on prediction ",fontdict=fontdict)
             plt.title('Pathologic',fontdict=fontdict)
             plt.legend(["Pathologic"],loc='best')
             st.pyplot()



