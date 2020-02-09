# Stay on Track!
My project **Stay on Track!** for [Insight Boston](https://www.insightdatascience.com/) Data Science 2020A predicts service interruptions of commuter rails using past alerts records from [MBTA](https://www.mbta.com/developers/v3-api/streaming), reliability & ridership metrics from
the [MBTA performance](https://mbta-massdot.opendata.arcgis.com/search?tags=mbta%2Ccommuter%20rail), and past climate records from [NOAA](https://www.noaa.gov/weather). All code associated with the [web app](https://stayontrack-1.herokuapp.com/) can be found here. 
This project the files for the web app that was made through Heroku and Streamlit. A light GBM model was used to estimate hours of service interruption of commuter rails for tomorrow 
using real-time weather forecast. The model is fast with a good performance. 


# Motivation
Most current apps alert commuters of the real-time arrival departure times of the trains. But commuter rail lines suffer several service interruptions en route that commuters cannot account for ahead of time. My app Stay on Track! informs commuters of interruptions for the trains running tomorrow. 
This would help commuters make informed decisions about their travel plans. 


Dependencies can be found in the environment.yml file, and this file can be used to create a conda environment with
```console
foo@bar:~$ conda env create -f environment.yml
```
