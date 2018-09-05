# Decoding functions
# Calc_MEG
# Pinheiro-Chagas - 2016

# Libraries
import sys
from GATclassifiers import Classification, LogisticRegression, NeuralNet
from initDirs import dirs
import numpy as np
import os

def Decoding(params, type, scorer, gatordiag):
    # Define firs if it' gat or only diagonal
    if gatordiag is 'diagonal':
        params['test_times'] = 'diagonal'

    if type == 'Classification':
        # Define scorer
        print('Decoding classification subject ' + params['subject'])
        scores = Classification(params['X_train'], params['y_train'], params['X_test'], params['y_test'], scorer, params['mode'], params)
    elif type == 'LogisticRegression':
        # Define scorer
        print('Decoding logistic regression subject ' + params['subject'])
        scores = LogisticRegression(params['X_train'], params['y_train'], params['X_test'], params['y_test'], scorer, params['mode'], params)
    elif type == 'NeuralNet':
        # Define scorer
        print('Decoding neural network subject ' + params['subject'])
        scores = NeuralNet(params['X_train'], params['y_train'], params['X_test'], params['y_test'], scorer, params['mode'], params)

    print('decoding subject ' + params['subject'] + ' done!')


    # Organize results
    results = ({'train_times': params['train_times'], 'test_times': params['test_times'], 'times': params['times'], 'scores': scores})
    print('results size is: ' + str(sys.getsizeof(results)) + ' bytes')

    # Save results
    print('saving results')
    save_dir = dirs['result'] + 'individual_results/' + params['train_set'] + '_' + params['test_set'] + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = save_dir + params['subject'] + '_' + params['train_set'] + '_' + params['test_set'] \
            + '_results_' + type + '_' + scorer + '_' + gatordiag + '_' + params['baseline_correction']
    np.save(fname, results)
    print('saving done')
    print(fname)

    return scores
