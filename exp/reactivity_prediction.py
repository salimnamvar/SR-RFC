""" Reactivity Prediction Experiment

    https://www.kaggle.com/code/jonasthoenfaber/reactivity-prediction
"""

import pandas as pd, numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model

# define numpy seed for reproducability
#np.random.seed(100)

train_filename = "G:/Challenges/RNA/data/train_data.csv"

# parameters before loading data
N_ROWS = 806_573 # from competation page (https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data?select=train_data.csv)
CHUNKSIZE = 200
SKIPROWS = np.random.randint(1, N_ROWS-CHUNKSIZE)

# load data
train_df = pd.read_csv(train_filename, skiprows=0, chunksize=1).get_chunk()
train_df = pd.read_csv(train_filename, skiprows=SKIPROWS, chunksize=CHUNKSIZE, names=train_df.keys()).get_chunk()

for i,j in zip(['A','G','U','C'], ['1','2','3','4']):
    train_df['sequence'] = train_df['sequence'].str.replace(i,j)
#train_df['sequence'] = train_df['sequence'].astype('uint8')

# load data (full data)
#train_filename = '/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv'
#train_df = pd.read_csv(train_filename)

react_list = []
react_err_list = []
for i,k in enumerate(train_df.keys()):
    if 'reactivity' in k and 'error' not in k:
        react_list.append(i)
    elif 'reactivity_error' in k:
        react_err_list.append(i)

#print(react_list)
#print(react_err_list)

inputs_length = 457
inputs = []
targets = []

# fill list with type of experiment (DMS_MaP or 2A3_MaP)
exp_type = []
exp_type_uniques = ['DMS_MaP', '2A3_MaP']

# loop through dataframe
for i, row in train_df.iterrows():

    # get reactivities
    r = np.array(row[react_list].values).astype('float')

    # get sequence
    seq = row.sequence  # .replace('A', '1').replace('G', '2').replace('U', '3').replace('C', '4')
    seq = np.array([*seq]).astype('int')

    # store input values
    # input = np.zeros(len(r)).astype('int')
    input = np.zeros(inputs_length).astype('int')
    input[:len(seq)] = seq

    # store target values
    target = np.zeros(inputs_length).astype('int')
    target[:len(r)] = np.nan_to_num(r).astype('int')
    wt = np.where(target)[0]
    if len(wt) > 0:
        target = np.roll(target, -wt[0])

    # clip target values between 0 and 1
    target[np.where(target < 0)] = 0
    target[np.where(target > 1)] = 1

    # append values
    exp_type.append(row.experiment_type)
    inputs.append(input)
    targets.append(target)

# convert lists to numpy arrays
exp_type = np.array(exp_type)
inputs = np.array(inputs)
targets = np.array(targets)

# parameters to tune models
EPOCHS = 1
BATCH_SIZE = 5
INPUT_SIZE = inputs.shape[1]
ADAM_VAL = 0.0001

# model on DMS maps
model_DMS = Sequential()
model_DMS.add(Dense(units=32, activation='relu', input_dim=INPUT_SIZE))
model_DMS.add(Dense(units=64, activation='relu'))
model_DMS.add(Dense(units=inputs.shape[1], activation='sigmoid'))
model_DMS.compile(optimizer=Adam(ADAM_VAL), loss=binary_crossentropy, metrics='mae')

# model on 2A3 maps
model_2A3 = Sequential()
model_2A3.add(Dense(units=32, activation='relu', input_dim=inputs.shape[1]))
model_2A3.add(Dense(units=64, activation='relu'))
model_2A3.add(Dense(units=inputs.shape[1], activation='sigmoid'))
model_2A3.compile(optimizer=Adam(ADAM_VAL), loss=binary_crossentropy,text metrics='mae')

# fit models
where_DMS = np.where(exp_type == exp_type_uniques[0])[0]
if len(where_DMS) > 0:
    print(f'Training DMS model with {len(where_DMS)}/{train_df.shape[0]} datapoints.')
    model_DMS.fit(x=inputs[where_DMS], y=targets[where_DMS], epochs=EPOCHS, batch_size=BATCH_SIZE)

where_2A3 = np.where(exp_type == exp_type_uniques[1])[0]
if len(where_2A3) > 0:
    print(f'Training 2A3 model with {len(where_2A3)}/{train_df.shape[0]} datapoints.')
    model_2A3.fit(x=inputs[where_2A3], y=targets[where_2A3], epochs=EPOCHS, batch_size=BATCH_SIZE)

model_DMS.save('model_DMS.keras')
model_2A3.save('model_2A3.keras')
del train_df, exp_type, inputs, targets, model_DMS, model_2A3


model_DMS = load_model('model_DMS.keras')
model_2A3 = load_model('model_2A3.keras')

test_filename = 'G:/Challenges/RNA/data/test_sequences.csv'
test_df = pd.read_csv(test_filename)

# make input numerixc
for i,j in zip(['A','G','U','C'], ['1','2','3','4']):
    test_df['sequence'] = test_df['sequence'].str.replace(i,j)

import time

t0, t00 = time.time(), time.time()

CHUNKSIZE = 2_000
EST_TOT = 269_796_671

# define dataframe with predictions
pred_df = pd.DataFrame(columns=['id', 'reactivity_DMS_MaP', 'reactivity_2A3_MaP'])

X = np.zeros((0, INPUT_SIZE)).astype('float16')
X_temp = np.zeros((1, INPUT_SIZE)).astype('float16')
id_extr = []

I, II = 0, 0
II_end = 5

# loop through dataframe
for i, row in test_df.iterrows():

    # get numeric inputs
    seq = row.sequence
    seq = [*seq]

    # fill input values
    X_temp = 0 * X_temp
    X_temp[0, :len(seq)] = np.array(seq).astype('int')
    X = np.append(X, X_temp, axis=0)
    id_extr.append([row.id_min, row.id_max + 1])

    if not (i + 1) % CHUNKSIZE:
        I += CHUNKSIZE * len(seq)
        II += 1
        print("%i %% : %i / %i" % (100 * I / EST_TOT, I, EST_TOT))  # , end='\r')

        if II == II_end:

            # do predictions
            p_DMS = model_DMS.predict(X)
            p_2A3 = model_2A3.predict(X)

            # add predictions to dataframe
            for j in range(len(id_extr)):
                ids = np.arange(id_extr[j][0], id_extr[j][1])
                df = pd.DataFrame({'id': ids,
                                   'reactivity_DMS_MaP': p_DMS[j][:len(ids)],
                                   'reactivity_2A3_MaP': p_2A3[j][:len(ids)]})
                pred_df = pd.concat([pred_df, df])

            # make id the index column
            pred_df = pred_df.set_index('id')

            # save predictions
            if i + 1 == II * CHUNKSIZE:
                print('New predict_submission_new.csv file...')
                pred_df.to_csv('predict_submission_new.csv', header=pred_df.keys())
            else:
                print('Appending to predict_submission_new.csv file...')
                pred_df.to_csv('predict_submission_new.csv', mode='a', header=False)

            print(pred_df.shape)
            print('Total/interval Time used: %.1f/ %.1f s' % (time.time() - t00, time.time() - t0))
            t0 = time.time()
            print()

            del X, pred_df, id_extr
            X = np.zeros((0, INPUT_SIZE)).astype('float16')
            pred_df = pd.DataFrame(columns=['id', 'reactivity_DMS_MaP', 'reactivity_2A3_MaP'])
            id_extr = []
            II = 0

if II != II_end:
    p_DMS = model_DMS.predict(X)
    p_2A3 = model_2A3.predict(X)

    # add predictions to dataframe
    for j in range(len(id_extr)):
        ids = np.arange(id_extr[j][0], id_extr[j][1])
        df = pd.DataFrame({'id': ids,
                           'reactivity_DMS_MaP': p_DMS[j][:len(ids)],
                           'reactivity_2A3_MaP': p_2A3[j][:len(ids)]})
        pred_df = pd.concat([pred_df, df])

    # make id the index column
    pred_df = pred_df.set_index('id')

    # save predictions
    print('Appending to predict_submission_new.csv file...')
    pred_df.to_csv('predict_submission_new.csv', mode='a', header=False)

print('DONE!!')