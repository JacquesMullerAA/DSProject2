# Disaster Response Pipeline Project


### Table of Contents

1. [Instructions](#instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `cd app`
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Motivation<a name="motivation"></a>

The project was completed to evaluate disaster response data, and to train a model from actual disaster response data that can analyse a message and then categorise it according to disaster categories. The model trained is applied in a web app that can categorise an entered message, as well as display plots evaluating the data.


## File Descriptions <a name="files"></a>

The files for this analysis are located [here](https://github.com/JacquesMullerAA/DSProject2)


## Results<a name="results"></a>

The results obtained as the two plots on the web app main page:
1. Distribution of infrastructure related messages genres (../img/infrastructure_genres.png):
    The distribution by genre of the messages that are all categorised as being infrastructure-related. Over 70% of the infrastructure-related messages fall in the "news" genre, followed by smaller percentages in the "direct" and "social" genres.

2. Top 15 disasted categories distribution of messages (../img/disaster_categories.png):
    The distribution of all messages by disaster category, shown for the top 15 categories. Most of the messages (almost 80%) are categorised as "related", followed then by "aid related", "weather related", "direct report", etc.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Actual disaster response data from Figure Eight (https://www.figure-eight.com/)
