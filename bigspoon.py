import streamlit as st
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


# Function to load model
@st.cache # cache
def load_model(modelName):
	#model = pd.read_pickle(os.path.join(folder, 'models', modelName + '.pkl'))
    model = pd.read_pickle(str('./models' + modelName + '.pkl'))
	#model = pd.read_pickle(str(modelName + '.pkl'))    
	return model

####how to write an address pd.read_pickle(str('./models' + modelName + '.pkl'))

# Load Model
modelName='Commuter_basic_linear_fit'
model = load_model(modelName)

dataName = 'REAL_DATA_v2'
data = load_model(dataName)


st.title('Cracked Tracks')


##pkl_filename = 'C:/Users/ritwi/Documents/GitHub/commuterrail/REAL_DATA_v2.pkl'
##with open(pkl_filename, 'rb') as file:
##    loaded_data = pickle.load(file)
#st.write(loaded_data)


user_input = st.selectbox('Which commuter rail?', data['Trains'].unique())
#user_input = st.text_input("Which commuter rail interests you? Enter a location like 'CR-Lowell', 'CR-Framingham'")
st.write(user_input)

print(data)
Data = pd.DataFrame(data)

#user_input2=st.selectbox('Which commuter rail?', Data['Trains'].unique())
#st.write(user_input2)


Train_df = pd.DataFrame()
for train in set(Data["Trains"]):
  print(train)
  if user_input == train:
    mask = Data["Trains"] == train
    Train_df = Data[mask]
st.write(Train_df.round(2))
#st.write(Train_df)

train_dict = {}
train_dict["Reliability"] = list()
train_dict["Frequency"] = list()
train_dict["Peak"] = list()
train_dict["Lag_Total_Hours_of_Service_Interruption"] =list()

yearmask = Train_df['Year'] == 2019
monthmask = Train_df['Month'] == 11
mask1 = (yearmask) & (monthmask) 
df = Train_df[mask]
train_dict["Reliability"].append(df['Reliability'])
train_dict["Frequency"].append(df['Frequency'])
train_dict["Peak"].append(df['Peak'])
train_dict["Lag_Total_Hours_of_Service_Interruption"].append(df['Lag'])

output = pd.DataFrame(df['Reliability'], df['Frequency'], df['Peak'], df['Lag'])
print(output)

#pkl_filename = 'C:/Users/ritwi/Documents/GitHub/commuterrail/Commuter_basic_linear_fit.pkl'
#with open(pkl_filename, 'rb') as file:
#    pickle_model = pickle.load(file)


prediction = model.predict(np.array([Train_df.iloc[-1,:].Reliability,Train_df.iloc[-1,:].Frequency,Train_df.iloc[-1,:].Peak,Train_df.iloc[-1,:].Lag]).reshape(1,4))
st.write(f"{user_input} is going to be down for {round(prediction.item(0),2)} hrs tomorrow.")

#month = pd.to_datetime(prediction, format = '%j').month
#day = pd.to_datetime(prediction, format = '%j').day
#st.write(calendar.month_name[month], day, user_input2)
