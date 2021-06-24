import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import bz2

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,classification_report


def load_data(database_filepath):
    """
    Load the merged messages and categories data
    
    Input:
        database_filename: string. Filename of the database where the data is saved in
    Output:
        X: dataframe. Messages data
        Y: dataframe. Categories data for each message
        category_names: List of the disaster categories
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages_data', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Normalize, tokenize and lemmatize a text string
    
    Input:
        text: string. Text that contains the message for processing
       
    Returns:
        tokens: List of normalized and lemmatized word tokens
    """

    # Detect and replace URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize, tokenize and remove punctuation
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    """
    Build the model pipeline and grid search cv
            
    Returns:
        model_cv: sklearn.model_selection.GridSearchCV. The model as a sklearn grid search cv
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize))
        , ('tfidf', TfidfTransformer())
        , ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {'vect__min_df': [5, 10], #1,5,10 - best = 10
                'vect__max_df': [0.5, 1.0], #0.5, 1.0 - best = 0.5
                'tfidf__use_idf':[True, False], #true,false - best=false
                'clf__estimator__n_estimators':[25, 50], #10,25,50 - best = 50
                'clf__estimator__min_samples_split':[5, 10]} #2,5,10 - best = 10

    model_cv = GridSearchCV(estimator=pipeline, param_grid=parameters,cv=3, verbose=3)
    return model_cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model on the test data
    
    Input:
        model: sklearn.model_selection.GridSearchCV. The model as a sklearn grid search cv
        X_test: dataframe. Messages test data
        Y_test: dataframe. Categories test data for each message
        category_names: List of the disaster categories
    """
    Y_pred = model.predict(X_test)
    
    for i in range(len(Y_test.columns)):
        print('Category: {} '.format(Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))
        print('F1 {}\n\n'.format(f1_score(Y_test.iloc[:, i].values, Y_pred[:, i],average='weighted')))

    print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = Y_test.columns[:-1]))
    

def save_model(model, model_filepath):
    """
    Save the model to a file with pickle
    
    Input:
        model: sklearn.model_selection.GridSearchCV. The model as a sklearn grid search cv
        model_filepath: string. The path of the file to save the model in
    """
    #pickle.dump(model, open(model_filepath, 'wb'))
    
    pickle.dump(model, bz2.BZ2File(model_filepath, 'w'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

    # To run file: python models/train_classifier.py data/DisasterResponse.db models/classifier.pbz2
                