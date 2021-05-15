# Disaster Response Pipeline Project
## Scope

The goal is to build a nlp pipeline that classifies disaster message into different categories. The project is divided into 3 parts:

 - The first part is an etl  pipeline that first loads the data from .csv files, 
preprocess them and creates a [sql lite](https://www.sqlite.org/index.html) data base.

- The second part loads the data from the data base and trains a model with the data. The model will be saved as pickle file. 

- The third part is a web-app. The web app is based on flask. The app includes plotly visualisations of the training data
    and the possibility to query the trained model with own requests. 


## Start the web application:
1. Setup anaconda environment: 
    Download [anaconda](https://www.anaconda.com/) and run:
    `conda env create -f environment.yml`
    


2. Preprocess data, extract entities in texts and create data base:
        
 (make sure that phyton path is the project directory `set PYTHONPATH=.` in windows or `export PYTHONPATH=.` in linux )
        
        
   
    `python data/process_data.py --messages_filepath <file path to messages.csv with texts> 
            --categories_filepath <file to categories.csv with lables for each message> 
                --database_filepath <file path for reult db>`
        

3. Train the ML model and save the model as pickle:

        `python models/train_classifier.py --model_filepath <Path to save the trained model> --database_filepath <file path to training data> 
                       --save_results <whether to save prediction results>  --model_name <name of the model. Has to be difined in the model_factory.py >
                        `

4. Run the following command in the app's directory (make sure that app directory is the python path `set PYTHONPATH=.`) to run your web app.
    `python run.py`

5. Access the application with the url: http://0.0.0.0:3001/


## Further Work: 

1) Add a [MongoDb](https://www.mongodb.com/) that saves send query messages and their predictions in order to label them later and 
   keep track of requested informations.
2) Split model request and applikation. This will make it easier to change the model and model setup without changes in the frontend application. 
3) Use a more sophisticated model to reach better accuracy. The class distributions are very unbalanced, this results in a low f1 score with the current model. 


