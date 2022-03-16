# Link predicition in a citation network

## Environment setting-up

```
pyenv virtualenv 3.9.9 link-prediction
pyenv local link-prediction
pip install -r requirements.txt
```

## Run experiments

The main script can be called this way: 

```python -m src.run_experiment --OPTIONS```

Script usage:

```
usage: run_experiment.py [-h] [--metadata_features | --no-metadata_features]
                         [--graph_features | --no-graph_features]
                         [--graph_learned_features | --no-graph_learned_features]
                         [--tfidf | --no-tfidf] [--experiment EXPERIMENT]
                         [--classification {nn,lr,rf,xgboost,svm}]
                         [--submission_name SUBMISSION_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --metadata_features, --no-metadata_features
                        Add the option for using metadata features (authors,
                        years, title) (default: False)
  --graph_features, --no-graph_features
                        Add the option for using basic graph features
                        (default: False)
  --graph_learned_features, --no-graph_learned_features
                        Add the option for using advanced learned graph
                        features with Word2Vec (default: False)
  --tfidf, --no-tfidf   Add the option for using advanced textual information
                        from the abstract with TFIDF (default: False)
  --experiment EXPERIMENT
                        Choose the experiment
  --classification {nn,lr,rf,xgboost,svm}
                        Choose a classifier
```