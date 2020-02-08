# Predicting service interruption of commuter rails in Greater Boston

This project details the collection of past alerts information from MBTA API (https://www.mbta.com/developers/v3-api/streaming), reliability & ridership metrics from
the MBTA performance website, and past climate records from NOAA. The information was combined to predict service interruption durations into tomorrow by incorporating 
tree-based learning algorithms. The model was used to create a web app through Heroku and Streamlit (check commuterrail repository). 

For the MBTA information, you would require the API key. The file has been sectioned into sub-parts including data collection, cleaning, and exploring the data, followed by running regression
models. The past alerts information included information from Nov 2017 to 2019 and contained service alert types such as track changes, delays, repairs, and maintenance. The description of the 
issues were cleaned and sorted. 
Furthermore, this information was combined temporally with reliability and ridership metrics of the 12 commuter lines. The frequency of the movement of the train 
was added from the commuter rail lines schedules. Direction of the train was extrapolated from the train ID information in the API. Imbalances in the data was handled by appropriate 
binning in 2 hour intervals across the day to be able to predict interruptions in every 2 hrs. 



