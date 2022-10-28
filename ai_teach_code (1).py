'''Copyright (c) 2022 AIClub

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without 
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of 
the Software, and to permit persons to whom the Software is furnished to do so, subject to the following 
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO 
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE 
OR OTHER DEALINGS IN THE SOFTWARE.

Follow our courses - https://www.corp.aiclub.world/courses'''

def launch_fe(data):
    import os
    import pandas as pd
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    MAX_TEXT_FEATURES = 200
    columns_list = ["category", "clean_text"]

    dataset = pd.read_csv(data, skipinitialspace=True)
    num_samples = len(dataset)

    label = "category"
    dataset = dataset.dropna(subset=[label])

    # Fill values missing in text features
    text_model_impute = \
        SimpleImputer(strategy='constant', fill_value='missing').fit(dataset[["clean_text"]])
    # Save the model
    model_name = "c8bba82d-871a-4b44-80af-97a31ab60353"
    fh = open(model_name, "wb")
    pickle.dump(text_model_impute, fh)
    fh.close()

    text_features = ["clean_text"]
    dataset[text_features] = text_model_impute.transform(dataset[text_features]) 

    # Encode text into numbers.
    # Encode one text column at a time.
    text_model = []
    for text_feature in ["clean_text"]:
        model = TfidfVectorizer(stop_words='english',
                                max_df=int(num_samples/2),
                                max_features=MAX_TEXT_FEATURES,
                                decode_error='ignore').fit(dataset[text_feature])
        text_model.append(model)
    # Save the model
    model_name = "e1ef44fe-ecec-4c03-9cbf-b5bb5bb401b4"
    fh = open(model_name, "wb")
    pickle.dump(text_model, fh)
    fh.close()

    for model, feature in zip(text_model, ["clean_text"]):
        data = model.transform(dataset[feature])
        new_feature_names = model.get_feature_names()
        new_feature_names = [feature + '_' + i for i in new_feature_names]
        if (sparse.issparse(data)):
            data = data.toarray()
        dataframe = pd.DataFrame(data, columns=new_feature_names)
        dataset = dataset.drop(feature, axis=1)
        # reset_index to re-order the index of the new dataframe.
        dataset = pd.concat([dataset.reset_index(drop=True), dataframe.reset_index(drop=True)], axis=1)

    # Encode labels into numbers starting with 0
    label = "category"
    tmpCol = dataset[label].astype('category')
    dict_encoding = { label: dict(enumerate(tmpCol.cat.categories))}
    # Save the model
    model_name = "5c98692c-fe50-4cae-a196-441373f3787f"
    fh = open(model_name, "wb")
    pickle.dump(dict_encoding, fh)
    fh.close()

    label = "category"
    dataset[label] = tmpCol.cat.codes

    # Move the label column
    cols = list(dataset.columns)
    colIdx = dataset.columns.get_loc("category")
    # Do nothing if the label is in the 0th position
    # Otherwise, change the order of columns to move label to 0th position
    if colIdx != 0:
        cols = cols[colIdx:colIdx+1] + cols[0:colIdx] + cols[colIdx+1:]
        dataset = dataset[cols]

    # split dataset into train and test
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Write train and test csv
    train.to_csv('train.csv', index=False, header=False)
    test.to_csv('test.csv', index=False, header=False)
    column_names = list(train.columns)
def get_model_id():
    return "5c98692c-fe50-4cae-a196-441373f3787f"

# Please replace the brackets below with the location of your data file
data = '<>'

launch_fe(data)

# import the library of the algorithm
from sklearn.neural_network import MLPClassifier

# Initialize hyperparams
hidden_layer_sizes = (500,)
activation = 'relu'
solver = 'adam'
alpha = 0.0001
learning_rate = 'constant'
learning_rate_init = 0.001
max_iter = 200
early_stopping = True

# Initialize the algorithm
model = MLPClassifier(random_state=1, hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter, early_stopping=early_stopping)

import pandas as pd
# Load the test and train datasets
train = pd.read_csv('train.csv', skipinitialspace=True, header=None)
test = pd.read_csv('test.csv', skipinitialspace=True, header=None)
# Train the algorithm
model.fit(train.iloc[:,1:], train.iloc[:,0])
def encode_confusion_matrix(confusion_matrix):
    import pickle
    encoded_matrix = dict()
    object_name = get_model_id()
    file_name = open(object_name, 'rb')
    dict_encoding = pickle.load(file_name)
    labels = list(dict_encoding.values())[0]
    for row_indx, row in enumerate(confusion_matrix):
        encoded_matrix[labels[row_indx]] = {}
        for item_indx, item in enumerate(row):
            encoded_matrix[labels[row_indx]][labels[item_indx]] = item
    return encoded_matrix

# Predict the class labels
y_pred = model.predict(test.iloc[:,1:])
# import the library to calculate confusion_matrix
from sklearn.metrics import confusion_matrix
# calculate confusion matrix
confusion_matrix = confusion_matrix(test.iloc[:,0], y_pred)
encoded_matrix = encode_confusion_matrix(confusion_matrix)
print('Confusion matrix of the model is: ', encoded_matrix)
# calculate accuracy
score = model.score(test.iloc[:, 1:], test.iloc[:, 0])
# The value is returned as a decimal value between 0 and 1
# converting to percentage
accuracy = score * 100
print('Accuracy of the model is: ', accuracy)

# fe_transform function traansforms raw data into a form the model can consume
print('Below is the prediction stage of the AI')
def fe_transform(data_dict, object_path=None):
    import os
    import pandas as pd
    from io import StringIO
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_extraction import text
    import pickle
    from scipy import sparse
    
    dataset = pd.DataFrame([data_dict])

    text_feature = ["clean_text"]
    object_name = "e1ef44fe-ecec-4c03-9cbf-b5bb5bb401b4"
    file_name = open(object_name, 'rb')
    text_model   = pickle.load(file_name)
    for model, feature in zip(text_model, text_feature):
        data = model.transform(dataset[feature])
        new_feature_names = model.get_feature_names()
        new_feature_names = [feature + '_' + i for i in new_feature_names]
        if (sparse.issparse(data)):
            data = data.toarray()
        dataframe = pd.DataFrame(data, columns=new_feature_names)
        dataset = dataset.drop(feature, axis=1)
        dataset = pd.concat([dataset, dataframe], axis=1)

    return dataset
def encode_label_transform_predict(prediction):
    import pickle
    encoded_prediction = prediction
    label = "category"
    object_name = "5c98692c-fe50-4cae-a196-441373f3787f"
    file_name = open(object_name, 'rb')
    dict_encoding = pickle.load(file_name)
    label_name = list(dict_encoding.keys())[0]
    encoded_prediction = \
        dict_encoding[label_name][int(prediction)]
    print(encoded_prediction)
def get_labels(object_path=None):
    label_names = []
    label_name = list(dict_encoding.keys())[0]
    label_values_dict = dict_encoding[label_name]
    for key, value in label_values_dict.items():
        label_names.append(str(value))

test_sample = {'clean_text': 'Hello'}
# Call FE on test_sample
test_sample_modified = fe_transform(test_sample)
# Make a prediction
prediction = model.predict(test_sample_modified)
encode_label_transform_predict(prediction)
