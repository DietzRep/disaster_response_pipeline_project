# Disaster Response Pipeline Project

### Instructions:
1. Setup anaconda environment: 
    Download [anaconda](https://www.anaconda.com/) and run:
    `conda env create -f environment.yml`
    
2. Preprocess data, extract entities in texts and create data base:

        `python data/process_data.py --messages_filepath <file path to messages.csv with texts> 
            --categories_filepath <file to categories.csv with lables for each message> 
                --database_filepath <file path for reult db>`
        

3. Train the ML model and save the model as pickle:

        `python models/train_classifier.py --model_filepath <Path to save the trained model> --database_filepath <file path to training data> 
                       --save_results <whether to save prediction results>  --model_name <name of the model. Has to be difined in the model_factory.py >
                        `

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
