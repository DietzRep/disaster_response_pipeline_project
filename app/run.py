import json
import plotly
import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from disaster_response_pipeline_project.data.column_utils import MESSAGES_COLUMNS, IGNORE_LABELS
from disaster_response_pipeline_project.models.model_factory import load_model
from disaster_response_pipeline_project.data.entity_extractor import EntityExtractor
import logging
import datetime

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/test.db')
data = pd.read_sql_query('SELECT name from sqlite_master where type= "table";', engine)
df = pd.read_sql_table('disaster_response', engine)
entity_extractor = EntityExtractor()
entities = pd.read_sql_table('text_entities', engine)
gpe_entities = entities[entities["labels"] == "GPE"].copy()
# load model
model = load_model("../models/", "random_forest.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    logging.info("Time: {0}, Request: Index".format(datetime.datetime.now()))
    # extract data needed for visuals
    location_counts = gpe_entities.entity.value_counts()
    location_names = list(location_counts.index)
    request_counts = df\
        .drop(MESSAGES_COLUMNS + IGNORE_LABELS, axis= 1).sum().sort_values(ascending=False).iloc[0:15]
    request_names = list(request_counts.index)
    graphs = [
        {
            'data': [
                Bar(
                    x=[str(x) for x in location_names][0:15],
                    y=[float(x) for x in location_counts.values][0:15]
                )
            ],

            'layout': {
                'title': 'most common locations',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "location"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[str(x).replace("_" , " ") for x in request_names][0:15],
                    y=[float(x) for x in request_counts.values][0:15]
                )
            ],

            'layout': {
                'title': 'most common requests',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "request"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    logging.info("Time: {0}, Request: Query".format(datetime.datetime.now()))
    # save user input in query
    sentence = request.args.get('query', '')
    query = np.array([[sentence, "direct"]])
    ents = entity_extractor.get_doc_entities(sentence)
    location = "No location provided"
    for ent in ents:
        if ent.label_ == "GPE":
            location = ent.text

    # use model to predict classification for query
    classification_labels = model.predict(query)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=sentence,
        location = location,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()