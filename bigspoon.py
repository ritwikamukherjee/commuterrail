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




if len(sys.argv) > 1:
    folder = os.path.abspath(sys.argv[1])
else:
    folder = os.path.abspath(os.getcwd())

#@st.cache
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
Light_GBM= 'Commuter_lightgbm' # 'Commuter_random_forest_regressor_trytry'      ###'Commuter_random_forest_regressor_trytry2';'Commuter_LightGBMClassifier_try1'    #'Commuter_random_forest_classifier2'
model = load_model(Light_GBM)
#model1=joblib.load(str(Random_forest + '.joblib'))
dataName = 'REAL_DATA_v6'
data = load_model(dataName)


st.title('Stay on Track!')


#Windows of peak hours: 
x1=datetime.time(7,00,00)
x2=datetime.time(9,00,00)
x3=datetime.time(16,00,00)
x4=datetime.time(19,00,00)
#########

train_input = st.selectbox("Choose your commuter rail", data['Trains'].unique())

st.header(f"So, you are traveling on {train_input} tomorrow.")
st.write("What's the time and direction of your travel?")
            
time_input = st.radio("Choose your time of travel:", ['12:00 AM', '2:00 AM', '4:00 AM', '6:00 AM', '8:00 AM', '10:00 AM', '12:00 PM', '2:00 PM',
                                                        '4:00 PM', '6:00 PM','8:00 PM', '10:00 PM', '11:00 PM'])
direction_input = st.radio("Choose your direction of travel:", ['Inbound', 'Outbound'])




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
                if key == [1]:
                    dict_others[key] = [1]
                if key == Day:
                    dict_others[key] = [1] 
            Reliability = Train_df.iloc[-1,:].Reliability
            Frequency = Train_df.iloc[-1,:].Frequency
            Temperature = 38
            Snow = 0
            Wind =7
            Prcp = 20
            Ridership_2018 = Train_df.iloc[-1,:].Ridership_2018
            Lag = Train_df.iloc[-1,:].Lag
            Snowlag = 0
           
            if (x1.hour <= travel_time.hour <= x2.hour) or (x3.hour <= travel_time.hour <= x4.hour):
                Peak = 1
            else:
                Peak = 0
            if direction_input == 'Outbound':
                Outbound = 1
                Inbound = 0
            if direction_input == 'Inbound':
                Outbound = 0
                Inbound = 1
            new_df1 = pd.DataFrame(dict_trains)
            new_df2 = pd.DataFrame(dict_others)            
            new_df = pd.concat([new_df1, new_df2], axis=1)
            #st.write(new_df)
            feature1= np.array([Day]).ravel()
            feature2=np.array([Hour]).ravel()
            features3 = np.array([Reliability, Frequency,Peak,Outbound, Inbound, Temperature, Snow, Wind, Prcp, Ridership_2018, Lag, Snowlag]).reshape(1,12)
            features4 = new_df.iloc[0,:].values.reshape(1,34)
            features = np.concatenate((feature1, feature2,features3,features4), axis = None).reshape(1,48)
            

            prediction = model.predict(features)
            output =(prediction.item(0)) * 60
            time_axis = np.array([0, 4, 8, 12, 16, 20, 24])
            bin_x = np.digitize(Hour,time_axis)
            
            t1 = time_axis[bin_x-1]
            t2 = time_axis[bin_x-1]+4
           
            d1 = datetime.datetime.strptime(f"{t1}:00", "%H:%M")
            d11 = d1.strftime("%I:%M %p")
            if t2 != 24:
                d2 = datetime.datetime.strptime(f"{t2}:00", "%H:%M")
                d21 = d2.strftime("%I:%M %p")
            else: 
                d2 = datetime.datetime.strptime(f"0:00", "%H:%M")
                d21 = d2.strftime("%I:%M %p")
            
            
            st.write(f"Based on historical data of service alerts, weather, and more recent repairs, {train_input}, may have service interruptions for {round(output, 2)} minutes between {d11} and {d21}, tomorrow.")
                            
            #time_axis = pd.DataFrame({time_axis)
            
            y_axis = list()
            ticklist = list()
            error = list()
            for i in range(0, len(time_axis)-1):
                ticklist.append(f"{time_axis[i]+2}:00")    #(f"{i*4}:00:00")  #(f"{time_axis[i]}:00:00")
                feature_hour = time_axis[i] #numpy array
                #st.write((feature_hour))
                features_hour = np.concatenate((feature1, feature_hour,features3,features4), axis = None).reshape(1,48)
                pred = model.predict(features_hour)
                y_axis.append(pred.item(0)*60)
                y_axis2 = np.array(y_axis)
                error.append(0.132*60)
                error2 = np.array(error)
                #st.write(len(y_axis2.flatten()))
            #st.write(y_axis2)
            #fig, ax = plt.subplots()
            #ax.plot(time_axis, y_axis2)
            ##ax.set_xlim([0,25])
            #ax.set_ylim([0,120])
            #ax.set_xticklabels(ticklist)
            #ax.set_xlabel("Time of day")
            #ax.set_ylabel("Duration of service interruption (min)") 
            ticklist_ampm = list()
            
            # for i in range(0, len(ticklist)):
                # f1= datetime.datetime.strptime(f"{ticklist[i]}", "%H:%M")
                # ticklist_ampm.append(f1.strftime("%I:%M %p"))
                # ticklist_ampm2 = np.array(ticklist_ampm)
           
            source = pd.DataFrame({'Time': ticklist, 'Estimated service interruption (min)': y_axis2, 'ci' :error2, 'ci1' : y_axis2-error2, 'ci2' :y_axis2+error2})
            bars = alt.Chart(source).mark_bar().encode(x= alt.X('Time', sort=None),y='Estimated service interruption (min)', color = alt.condition(
                                alt.datum.Time == f"{time_axis[bin_x-1]+2}:00", alt.value('green'), alt.value('grey'))).properties(width=500,height=400)
  
            error_bars = alt.Chart(source).mark_errorbar(extent = 'ci').encode(x= alt.X('Time', sort=None),y = 'Estimated service interruption (min)')

            chart = (bars + error_bars).configure_axis(labelFontSize=15, titleFontSize=15)#.facet(column='site:N'))   
            st.altair_chart(chart)
            #st.line_chart(source)
            #st.pyplot()
                
 
                
                            

                    ###Should do it for all hours and plot estimations
                    # st.write(sns.scatterplot(x = range(0,24), y = prediction.item(0)))                    
                    # chart_data = pd.DataFrame(np.random.randn(24, 3),columns=['a', 'b', 'c'])
                    # st.line_chart(chart_data)           
          
                                    



                            
                            
                            
                            
                            

#def visualize_data(df, x_axis, y_axis):
#    graph = alt.Chart(df).mark_circle(size=60).encode(
#        x=x_axis,
#        y=y_axis,
#        color='Origin',
#        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
#    ).interactive()

#   st.write(graph)
