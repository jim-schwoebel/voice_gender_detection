import numpy as np 
import json, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: Make sure that the class is labeled 'target' in the data file
g=json.load(open('males_females_audio_tpotclassifier_.json'))
tpot_data=g['labels']
features=g['data']

training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data, random_state=None)

# Average CV score on the training set was:0.9114324938848934
exported_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.01, dual=False, penalty="l2")
)

exported_pipeline.fit(training_features, training_target)
print('saving classifier to disk')
f=open('males_females_audio_tpotclassifier.pickle','wb')
pickle.dump(exported_pipeline,f)
f.close()
