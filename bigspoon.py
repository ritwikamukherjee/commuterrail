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

####how to write an address pd.read_pickle(str('./models' + modelName + '.pkl'))

# Load Model
modelName='Commuter_random_forest_regressor_trytry2'     #'Commuter_LightGBMClassifier_try1'    #'Commuter_random_forest_classifier2'
model = load_model(modelName)

dataName = 'REAL_DATA_v6'
data = load_model(dataName)


st.title('Stay on Track!')


##pkl_filename = 'C:/Users/ritwi/Documents/GitHub/commuterrail/REAL_DATA_v2.pkl'
##with open(pkl_filename, 'rb') as file:
##    loaded_data = pickle.load(file)
#st.write(loaded_data)


#user_input = st.sidebar.selectbox('Which commuter rail?', data['Trains'].unique())
#user_input = st.text_input("Which commuter rail interests you? Enter a location like 'CR-Lowell', 'CR-Framingham'")
#st.write(user_input)
#user_input 
#print(data)
Data = pd.DataFrame(data)
today = datetime.datetime.today()
my_date = date.today()
Weekday_name = calendar.day_name[my_date.weekday()]
#user_input2=st.selectbox('Which commuter rail?', Data['Trains'].unique())
#st.write(user_input2)


#month = pd.to_datetime(prediction, format = '%j').month
#day = pd.to_datetime(prediction, format = '%j').day
#st.write(calendar.month_name[month], day, user_input2)
dict_trains = {'CR-Fairmount':[0], 'CR-Fitchburg':[0], 'CR-Franklin':[0], 'CR:Greenbush' :[0], 'CR-Haverhill' :[0],
                                        'CR-Kingston' :[0], 'CR-Lowell':[0], 'CR-Lowell' :[0], 'CR-Middleborough':[0], 'CR-Needham':[0],
                                        'CR-Newburyport':[0], 'CR-Providence':[0], 'CR-Worcester':[0]}

dict_others = {'2017':[0], '2018':[0], '2019':[0],'1':[0], '2':[0], '3':[0], '4':[0],	'5':[0],	'6':[0],'7':[0], '8':[0],'9':[0], '10':[0],'11':[0],'12':[0], 'Friday':[0],
	'Monday':[0],	'Saturday':[0], 'Sunday':[0], 'Thursday':[0], 'Tuesday':[0], 'Wednesday':[0]}

#Windows of peak hours: 
x1=datetime.time(7,00,00)
x2=datetime.time(9,00,00)
x3=datetime.time(16,00,00)
x4=datetime.time(19,00,00)
#########


train_input = st.selectbox("Choose your commuter rail", data['Trains'].unique())

time_input = st.radio("Choose your time of travel:", ['12:00:00 AM', '2:00:00 AM', '4:00:00 AM', '6:00:00 AM', '8:00:00 AM', '10:00:00 AM', '12:00:00 PM', '2:00:00 PM',
                                                        '4:00:00 PM', '6:00:00 PM','8:00:00 PM', '10:00:00 PM'])
direction_input = st.radio("Choose your direction of travel:", ['Inbound', 'Outbound'])

lets_go = None
lets_go = st.button ("Go")

if lets_go is not None:
    Train_df = pd.DataFrame()
    Train_df = pd.DataFrame()
    for train in set(data['Trains']):
        if train_input == train:
            mask = Data["Trains"] == train
            Train_df = Data[mask]
            #st.write(Train_df)
            #if st.button('Click here'):
            st.header(f"So, you are traveling on {train_input} tomorrow.")
            st.write("What time were you thinking to travel?")
            #st.write(df)
            #peak_input = st.sidebar.selectbox("Choose peak/non-peak hour", ['Peak', 'Non-peak'])
            
            #time_input = st.sidebar.selectbox("Choose your time of travel", ['12:00:00 AM', '2:00:00 AM', '4:00:00 AM', '6:00:00 AM', '8:00:00 AM', '10:00:00 AM', '12:00:00 PM', '2:00:00 PM',
             #                                               '4:00:00 PM', '6:00:00 PM','8:00:00 PM', '10:00:00 PM'])                                              
            
            #direction_input = st.sidebar.selectbox("Choose inbound/outbound", ['Inbound', 'Outbound'])
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
            Temperature = 39.2
            Snow = 0
            Wind =5.75
            Prcp = 2
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
            features3 = np.array([Reliability, Frequency,Peak,Outbound, Inbound, Temperature, Snow, Wind,Prcp,Ridership_2018,Lag,Snowlag]).reshape(1,12)
            features4 = new_df.iloc[0,:].values.reshape(1,34)
            features = np.concatenate((feature1, feature2,features3,features4), axis = None).reshape(1,48)
            
            
            
            prediction = model.predict(features)
            output =(prediction.item(0)) * 60
                #mask=peakmask & outboundmask
                #df_of_interest = Train_df[mask] #this has all the other variables in case we want to plot something
            #st.write(today)
            #prediction = model.predict(np.array([Train_df.iloc[-1,:].Reliability,Train_df.iloc[-1,:].Frequency,Train_df.iloc[-1,:].Peak,Train_df.iloc[-1,:].Lag]).reshape(1,4))
            st.write(f"{train_input} is going to be down for {round(output, 2)} minutes tomorrow in the next four hours.")
            
            # time_axis = np.array([0, 4, 8, 12, 16, 20, 24])
            # #time_axis = pd.DataFrame({time_axis)
            # y_axis = list()
            # ticklist = list()
            # for i in range(0, len(time_axis)):
                # ticklist.append(f"{i*4}:00:00")
                # feature_hour = time_axis[i] #numpy array
                
                # #st.write((feature_hour))
                # features_hour = np.concatenate((feature1, feature_hour,features3,features4), axis = None).reshape(1,48)
                # pred = model.predict(features_hour)
                # y_axis.append(pred.item(0)*60)
                # y_axis2 = np.array(y_axis)
                # #st.write(len(y_axis2.flatten()))
            # #st.write(y_axis2)
            # #fig, ax = plt.subplots()
            # #ax.plot(time_axis, y_axis2)
            # ##ax.set_xlim([0,25])
            # #ax.set_ylim([0,120])
            # #ax.set_xticklabels(ticklist)
            # #ax.set_xlabel("Time of day")
            # #ax.set_ylabel("Duration of service interruption (min)") 
            
            # source = pd.DataFrame({'Hour of day': time_axis, 'Estimated service interruption (min)': y_axis2})
            # chart = alt.Chart(source).mark_line().encode(x='Hour of day',y='Estimated service interruption (min)').properties(width=900,
                        # height=500)
    

            # st.altair_chart(chart)
            # #st.line_chart(source)
            # st.pyplot()
                
                
                
                            

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
