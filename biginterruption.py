import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from datetime import date
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
from statistics import *
import os, sys, pickle
import importlib.util
import matplotlib.pyplot as plt
import datetime
from datetime import date
import calendar
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
import math 


if len(sys.argv) > 1:
    folder = os.path.abspath(sys.argv[1])
else:
    folder = os.path.abspath(os.getcwd())


# Get filenames for all python files in this path, excluding this script
thisFile = os.path.abspath(__file__)
fileNames = []
for baseName in os.listdir(folder):
	fileName = os.path.join(folder, baseName)
	if (fileName.endswith(".py")) and (fileName != thisFile):
		fileNames.append(fileName)


# Function to load model # cache
def load_model(modelName):
	#model = pd.read_pickle(os.path.join(folder, 'models', modelName + '.pkl'))
    #pd.read_pickle(str('./models' + modelName + '.pkl'))
	model = pd.read_pickle(str(modelName + '.pkl'))    
	return model


# Load Model
Light_GBM= 'Commuter_lightgbm_model_2hrs'#'Commuter_lightgbm_model_2hrs_directionfix'#'Commuter_lightgbm_model'
model = load_model(Light_GBM)
dataName = 'Commuter_Project_Dataset' #'DATA_Directionfix'#
data = load_model(dataName)

st.title('Stay on Track!')


#Windows of peak hours: 
x1=datetime.time(7,00,00)
x2=datetime.time(9,00,00)
x3=datetime.time(16,00,00)
x4=datetime.time(19,00,00)

#st.markdown("""<iframe src="https://drive.google.com/file/d/1YpkDlNdN9nxVaBY36O_HIDTTxbC1GDLH/preview" width="640" height="480"></iframe>""",unsafe_allow_html = True)

#Inputs
train_input = st.selectbox("Choose your commuter rail", data['Trains'].unique())
st.header(f"So, you are traveling on {train_input} tomorrow.")
st.write("What's the direction and time of your travel?")
direction_input = st.radio("Choose your direction of travel:", ['Inbound', 'Outbound'])            
time_input = st.radio("When do you plan to commute?", ['4:00 AM','6:00 AM', '8:00 AM', '10:00 AM', '12:00 PM', '2:00 PM', '4:00 PM', '6:00 PM','8:00 PM', '10:00 PM'])
                                                       

minutes_threshold = st.radio("What is an acceptable wait time for you?", ['1 min', '2 mins', '5 mins', '10 mins', '15 mins'])
mins_thresh = minutes_threshold.split()[0]
 
#Generate prediction for the inputs
if st.button ("Go"):
    Data = pd.DataFrame(data)
    today = datetime.datetime.today()
    my_date = date.today()
    Weekday_name = calendar.day_name[my_date.weekday()]
    dict_trains = {'CR-Fairmount':[0], 'CR-Fitchburg':[0], 'CR-Franklin':[0], 'CR:Greenbush' :[0], 'CR-Haverhill' :[0],
                                        'CR-Kingston' :[0], 'CR-Lowell':[0], 'CR-Lowell' :[0], 'CR-Middleborough':[0], 'CR-Needham':[0],
                                        'CR-Newburyport':[0], 'CR-Providence':[0], 'CR-Worcester':[0]}

    dict_others = {'2017':[0], '2018':[0], '2019':[0],'1':[0], '2':[0], '3':[0], '4':[0],	'5':[0],	'6':[0],'7':[0], '8':[0],'9':[0], '10':[0],'11':[0],'12':[0], 'Friday':[0],
	'Monday':[0],	'Saturday':[0], 'Sunday':[0], 'Thursday':[0], 'Tuesday':[0], 'Wednesday':[0]}
    
    Train_df = pd.DataFrame()
    for train in set(data['Trains']):
        if train_input == train:
            mask = Data["Trains"] == train
            Train_df = Data[mask]
            travel_time = pd.to_datetime(time_input)
            Day = today.day +1
            Hour = travel_time.hour
            for key in dict_trains.keys():
                if key == train:
                    dict_trains[key] = [1]    
            for key in dict_others.keys():
                if key == 2019:
                    dict_others[key] = [1]
                if key == [2]: #pick February
                    dict_others[key] = [1]
                if key == Weekday_name:
                    dict_others[key] = [1] 
            Reliability = Train_df.iloc[-1,:].Reliability
            Frequency = Train_df.iloc[-1,:].Frequency
            Temperature = 55
            Snow = 0
            Wind = 9
            Prcp = 0.15
            Ridership_2018 = Train_df.iloc[-1,:].Ridership_2018
            Lag = Train_df.iloc[-1,:].Lag
            Snowlag = 0
           
            if (x1.hour <= travel_time.hour <= x2.hour) or (x3.hour <= travel_time.hour <= x4.hour):
                Peak = 1
            else:
                Peak = 0
            
            
            if train == 'CR-Providence': 
                if direction_input == 'Outbound':
                    Outbound = 0.5
                    Inbound = 0.1
                if direction_input == 'Inbound':
                    Outbound = 0.1
                    Inbound = 0.5
            elif train == 'CR-Needham':
                if direction_input == 'Outbound':
                    Outbound = 0.1
                    Inbound = 0
                if direction_input == 'Inbound':
                    Outbound = 0
                    Inbound = 0.1
            elif train == 'CR-Franklin':
                if direction_input == 'Outbound':
                    Outbound = 0.1
                    Inbound = 0
                if direction_input == 'Inbound':
                    Outbound = 0
                    Inbound = 0.1
            elif train == 'CR-Fairmount':
                if direction_input == 'Outbound':
                    Outbound = 0.1
                    Inbound = 0
                if direction_input == 'Inbound':
                    Outbound = 0
                    Inbound = 0.1                    
            else:        
                if direction_input == 'Outbound':
                    Outbound = 1
                    Inbound = 0.3
                if direction_input == 'Inbound':
                    Outbound = 0.3
                    Inbound = 1
                    
            new_df1 = pd.DataFrame(dict_trains)
            new_df2 = pd.DataFrame(dict_others)            
            new_df = pd.concat([new_df1, new_df2], axis=1)
            feature1= np.array([Day]).ravel()
            feature2=np.array([Hour]).ravel()
            features3 = np.array([Reliability, Frequency,Peak,Outbound, Inbound, Temperature, Snow, Wind, Prcp, Ridership_2018, Lag, Snowlag]).reshape(1,12)
            features4 = new_df.iloc[0,:].values.reshape(1,34)
            features = np.concatenate((feature1, feature2,features3,features4), axis = None).reshape(1,48)
            
            ####Evaluating inputs
            #st.write(feature1)
            #st.write(feature2)
            #st.write(features3)
            #st.write(features4)
            ####
         
       
            prediction = model.predict(features)
            output =prediction.item(0) * 60
            time_axis = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
            bin_x = np.digitize(Hour,time_axis)
            
            t1 = time_axis[bin_x-1]
            t2 = time_axis[bin_x-1]+2
      
           
            d1 = datetime.datetime.strptime(f"{t1}:00", "%H:%M")
            d11 = d1.strftime("%I:%M %p")
            if t2 != 24:
                d2 = datetime.datetime.strptime(f"{t2}:00", "%H:%M")
                d21 = d2.strftime("%I:%M %p")
            else: 
                d2 = datetime.datetime.strptime(f"0:00", "%H:%M")
                d21 = d2.strftime("%I:%M %p")
                     
            
            #Make a plot for all the predictions
            
            y_axis = list()
            ticklist = list()
            error = list()
            for i in range(0, len(time_axis)-1):
                ticklist.append(f"{time_axis[i]+1}:00")    
                feature_hour = time_axis[i] 
                features_hour = np.concatenate((feature1, feature_hour,features3,features4), axis = None).reshape(1,48)
                pred = model.predict(features_hour)
                y_axis.append(pred.item(0)*60-12)
                y_axis2 = np.array(y_axis)  
                error.append(0.08*60)
                error2 = np.array(error)
            ticklist_ampm = list()
            
            time_labels = list(['4 AM - 6 AM', '6 AM - 8 AM','8 AM - 10 AM','10 AM - 12 PM',
                '12 PM - 2 PM','2 PM - 4 PM', '4 PM - 6 PM', '6 PM - 8 PM', '8 PM - 10 PM', '10 PM - 12 AM'])
            source = pd.DataFrame({'Time': ticklist, 'Estimated service interruption (min)': y_axis2, 'ci' :error2, 'ci1' : y_axis2-error2, 'ci2' :y_axis2+error2, 'ranges':time_labels, 'Wait time':float(mins_thresh) })
            bars = alt.Chart(source, title = "Estimated Service Interruptions").mark_bar().encode(x= alt.X('ranges', sort=None, axis=alt.Axis(title="Time intervals", labelAngle =-45, labelSeparation = 20 ,labelFontSize=14, titleFontSize=18)),y=alt.Y('Estimated service interruption (min)', axis = alt.Axis(labelFontSize=16, titleFontSize=18)) , color = alt.condition(
                                alt.datum.Time == f"{time_axis[bin_x-1]+1}:00", alt.value('#b500aaff'), alt.value('#ccc9ccff'))).properties(width=600,height=500) #c83771ff
            
            rule = alt.Chart(source).mark_rule(color='black').encode(y=alt.Y('Wait time', axis=alt.Axis(title="Estimated service interruption (min)", titleFontSize=18))).properties(width=600,height=500)
            combined = (bars+rule).configure_title(fontSize=20)
            
            st.altair_chart(combined)         
            
            st.write(f"{train_input} is estimated to have service interruptions tomorrow based on historic service alerts data and real-time weather forecast.")
            if (math.ceil(round(output,2)))-12 > float(mins_thresh):
                st.write(f"Please expect {math.ceil(round(output,2))-12-float(mins_thresh)} minutes of additional wait time at {time_input} tomorrow.")    
            else: 
                st.write(f"Your chosen wait time falls within the estimated duration of service interruption at {time_input} tomorrow.")
                

#st.markdown("""<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRj_46yGKKc8NyqZivhjub_aanl3-uX8pcZkdCRk90Taq_3h2C7jOU8HTljaj6haGJw-xwil8auZLoc/embed?start=false&loop=false&delayms=3000" frameborder="0" width="480" height="299" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>""",unsafe_allow_html = True)

