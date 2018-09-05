
# Packages
import mne
from ieegmat2mne import ieegmat2mne
import pandas as pd
import numpy as np


def prepDataDecoding(dirs, subject, project_name, train_set, test_set, baselinecorr, decimate):
    """ Prepare data to
    Example:
    :param dirs: taken from initDirs
    :param subject: 's01'
    :param project_name: UCLA, Calculia, etc
    :param train_set: 'op1'
    :param test_set: 'op2'
    :param baselinecorr: 'baseline_correct' or baseline_nocorrect
    :param decimate: 0 for no decimate (downsample)
    :return: params
    """

    # Preprocessing
    baseline = (-0.2, -0.05)  # time for the baseline period

    # Import epochs calc
    print('importing calc data')
    fname_data = dirs['data'] + 'data_all_' + subject + '_' + project_name + '.mat'
    fname_trialinfo = dirs['data'] + 'trialinfo_' + subject + '_' + project_name + '.csv'
    epochs, trialinfo = ieegmat2mne(fname_data, fname_trialinfo)
    epochs.decimate(decimate)
    print('done')

    # Baseline correct if needed
    if baselinecorr == 'baseline':
        print('baseline correcting')
        epochs.apply_baseline(baseline)
        print('done')


    # Select data
    print('selecting data')
    if train_set == test_set:  # Check whether we are training and testing on the same data
        mode = 'cross-validation'

        if project_name == 'UCLA':

            if train_set == 'calc_memory':
                train_index = trialinfo['isCalc'].notnull()
                X_train = epochs[train_index].get_data()
                y_train = np.array(trialinfo[train_index]['isCalc'])
                y_train = y_train.astype(np.float64)
                X_test = X_train
                y_test = y_train
                #train_times = {'start': 1.5, 'stop': 3.2}  # 'length': 0.05 defonce memory!
                train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
                test_times = train_times

            elif train_set == 'cres_group':
                trialinfo[trialinfo['corrResult'] == 4] = 3
                trialinfo[trialinfo['corrResult'] == 5] = 6
                train_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
                X_train = epochs[train_index].get_data()
                y_train = np.array(trialinfo[train_index]['corrResult'])
                y_train = y_train.astype(np.float64)
                X_test = X_train
                y_test = y_train
                train_times = {'start': -.2, 'stop': 4}  # 'length': 0.05 defonce memory!
                test_times = train_times

        elif project_name == 'Calculia_China':

            if train_set == 'operation':
                train_index = trialinfo['operator'] != 2
                X_train = epochs[train_index].get_data()
                y_train = np.array(trialinfo[train_index]['operator'])
                y_train = y_train.astype(np.float64)
                X_test = X_train
                y_test = y_train
                train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
                test_times = train_times

            if train_set == 'op2':
                train_index = trialinfo['operator'] != 2
                X_train = epochs[train_index].get_data()
                y_train = np.array(trialinfo[train_index]['op2'])
                y_train = y_train.astype(np.float64)
                X_test = X_train
                y_test = y_train
                train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
                test_times = train_times

        elif project_name == 'Number_comparison':

            if train_set == 'dot_digit':
                train_index = trialinfo['stim_type'] != 3
                X_train = epochs[train_index].get_data()
                y_train = np.array(trialinfo[train_index]['stim_type'])
                y_train = y_train.astype(np.float64)
                X_test = X_train
                y_test = y_train
                train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
                test_times = train_times

        elif project_name == 'MMR':

            if train_set == 'math_memory':
                # randomly select the same number of conditions memory from math
                math_trials = np.asanyarray(np.where(trialinfo['isCalc'] == 1))
                memory_trials = np.asanyarray(np.where(trialinfo['isCalc'] == 0))
                memory_trials_matched = np.random.choice(memory_trials.flatten(), np.size(math_trials))
                # concatenate balanced set
                train_index = np.sort(np.concatenate([memory_trials_matched.flatten(), math_trials.flatten()]))
                X_train = epochs[train_index].get_data()
                y_train = np.array(trialinfo.iloc[train_index]['isCalc'])
                y_train = y_train.astype(np.float64)
                X_test = X_train
                y_test = y_train
                train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
                test_times = train_times


        elif train_set == 'cres_sub_all':
            train_index = trialinfo['operator'] == -1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'pres':
            train_index = (trialinfo['presResult'] >= 3) & (trialinfo['presResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['presResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop':  3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'cres_riemann':
            epochs.pick_types(meg='grad')
            train_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs[train_index]
            X_train.crop(1.6, 3.2)
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'cres_riemann_add':
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] == 1
            X_train = epochs[train_index]
            X_train.crop(0, 3.2)
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'cres_riemann_sub':
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] == -1
            X_train = epochs[train_index]
            X_train.crop(0, 3.2)
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'absdeviant_riemann':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            epochs_reslock.pick_types(meg='grad')
            train_index = (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_train = epochs_reslock[train_index]
            X_train.crop(0, .8)
            y_train = np.array(trialinfo_reslock[train_index]['absdeviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'cres_aCSC':
            fname_calc = dirs['data'] + subject + '_calc_AICA_acc.mat'  # make this dynamic
            epochs, trialinfo = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            train_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs[train_index]
            epochs.pick_types(meg='grad')
            X_train.crop(1.6, 3.2)
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub_riemann':
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            X_train.crop(1.6, 3.2)
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': .8, 'stop': 1.6}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub_riemann_1600_3200':
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            X_train.crop(.2, 2.4)
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': .8, 'stop': 1.6}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'cres_len100ms':
            train_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2, 'length': 0.1}
            test_times = train_times

        elif train_set == 'cres_len200ms':
            train_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.5, 'stop': 3.2, 'length': 0.2, 'step': 0.05}
            test_times = train_times

        elif train_set == 'cres_alltimes':
            train_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 2.4}
            test_times = train_times

        elif train_set == 'cres_add':
            train_index = trialinfo['operator'] == 1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 4.4}
            test_times = train_times

        elif train_set == 'op2':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            train_times = {'start': 1.5, 'stop': 2}  # 'length': 0.05 defonce memory!

            test_times = train_times

        elif train_set == 'op2_add':
            train_index = trialinfo['operator'] == 1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'op2_sub':
            train_index = trialinfo['operator'] == -1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'op2_123':
            train_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] != 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'op2_len200ms':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.5, 'stop': 3.2, 'length': 0.2, 'step': 0.05}
            test_times = train_times

        elif train_set == 'op2_riemann':
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            X_train.crop(1.6, 3.2)
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.6, 'stop': 2.4, 'length': 0.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'op1':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'op1_len200ms':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2, 'length': 0.2}
            test_times = train_times

        elif train_set == 'op1_riemann':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            X_train.crop(0, 1.6)
            y_train = np.array(trialinfo[train_index]['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub_len200ms':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2, 'length': 0.2}
            test_times = train_times

        elif train_set == 'addsub_op2_0':
            train_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub_op2_1':
            train_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 1)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub_op2_2':
            train_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 2)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'addsub_op2_3':
            train_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 3)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -.2, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif train_set == 'op1_len50ms':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -0.1, 'stop': 1.6, 'length': 0.05}
            test_times = train_times
        elif train_set == 'op2_len50ms':
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': 1.5, 'stop': 3.2, 'length': 0.05}
            test_times = train_times
        elif train_set == 'vsa':
            train_index = info_vsa['congruency'] == 1
            X_train = epoch_vsa[train_index]
            y_train = np.array(info_vsa[train_index]['cue'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': -0.1, 'stop': 1.5}
            test_times = {'start': -0.1, 'stop': 1.5}
            # Response lock
        elif train_set == 'resplock_respside':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = trialinfo_resplock['accuracy'] == 1
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['respSide'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times
        elif train_set == 'resplock_choice':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['accuracy'] == 1) & (trialinfo_resplock['operator'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['correct_choice'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times
        elif train_set == 'resplock_correctness':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['accuracy'] == 1) & (trialinfo_resplock['operator'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['deviant'])
            y_train[y_train != 0] = 1
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_absdeviant':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['absdeviant'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['absdeviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_absdeviant_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['absdeviant'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['absdeviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times), 'length': 0.05}
            test_times = train_times

        elif train_set == 'resplock_deviant':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['deviant'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['deviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_deviant_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['deviant'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['deviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times),'length': 0.05}
            test_times = train_times

        elif train_set == 'resplock_cres':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['corrResult'] >= 3) & (trialinfo_resplock['corrResult'] <= 6) & (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['accuracy'] == 1)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_cres_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['corrResult'] >= 3) & (trialinfo_resplock['corrResult'] <= 6) & (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['accuracy'] == 1)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times), 'length': 0.05}
            test_times = train_times

        elif train_set == 'resplock_pres':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['presResult'] >= 3) & (trialinfo_resplock['presResult'] <= 6) & (trialinfo_resplock['operator'] != 0) & (trialinfo_resplock['accuracy'] == 1)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['presResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_op1':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['accuracy'] == 1) & (trialinfo_resplock['operator'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_op2':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['accuracy'] == 1) & (trialinfo_resplock['operator'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

        elif train_set == 'resplock_addsub':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresponse.mat'  # make this dynamic
            epochs_resplock, trialinfo_resplock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_resplock.decimate(decimate)
            train_index = (trialinfo_resplock['accuracy'] == 1) & (trialinfo_resplock['operator'] != 0)
            X_train = epochs_resplock[train_index]
            y_train = np.array(trialinfo_resplock[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_resplock.times), 'stop': np.max(epochs_resplock.times)}
            test_times = train_times

            ### Result lock
        elif train_set == 'resultlock_cres':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['accuracy'] == 1)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'resultlock_cres_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['accuracy'] == 1)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times), 'length': 0.2}
            test_times = train_times

        elif train_set == 'resultlock_pres':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['presResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times
            X_train_info = trialinfo_reslock[train_index]

        elif train_set == 'resultlock_pres_c':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] == 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['presResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times
            X_train_info = trialinfo_reslock[train_index]

        elif train_set == 'resultlock_pres_i':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['presResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times
            X_train_info = trialinfo_reslock[train_index]

        elif train_set == 'resultlock_pres_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['presResult'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times), 'length': 0.2}
            test_times = train_times

        elif train_set == 'resultlock_op1':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = trialinfo_reslock['operator'] != 0
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'resultlock_op2':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = trialinfo_reslock['operator'] != 0
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['operand2'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'resultlock_addsub':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = trialinfo_reslock['operator'] != 0
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'resultlock_absdeviant':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['absdeviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'resultlock_absdeviant_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['absdeviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times), 'length': 0.2}
            test_times = train_times

        elif train_set == 'resultlock_deviant':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['deviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times)}
            test_times = train_times

        elif train_set == 'resultlock_deviant_len200ms':
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'  # make this dynamic
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            train_index = (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['deviant'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            train_times = {'start': np.min(epochs_reslock.times), 'stop': np.max(epochs_reslock.times), 'length': 0.2}
            test_times = train_times

        elif train_set == 'vsa':
            train_index = info_vsa['congruency'] == 1
            info_vsa[info_vsa['cue'] == 1] = -1
            info_vsa[info_vsa['cue'] == 2] = 1
            X_train = epoch_vsa[train_index]
            y_train = np.array(info_vsa[train_index]['cue'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = y_train
            # Update params
            train_times = {'start': -0.1, 'stop': 1.5}
            test_times = train_times
    else:
        mode = 'mean-prediction'
        if (train_set == 'op1') & (test_set == 'resultlock_pres'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            #train_index = trialinfo['operand1'] != trialinfo['presResult']
            X_train = epochs
            y_train = np.array(trialinfo['operand1'])
            y_train = y_train.astype(np.float64)
            test_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            X_test = epochs_reslock[test_index]
            y_test = np.array(trialinfo_reslock[test_index]['presResult'])
            y_test = y_test.astype(np.float64)
            train_times = {'start': -0.2, 'stop': 0.8}
            test_times = {'start': -0.2, 'stop': 0.8}

        elif (train_set == 'op1') & (test_set == 'resultlock_pres_c'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            #train_index = trialinfo['operand1'] != trialinfo['presResult']
            X_train = epochs
            y_train = np.array(trialinfo['operand1'])
            y_train = y_train.astype(np.float64)
            test_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] == 0)
            X_test = epochs_reslock[test_index]
            y_test = np.array(trialinfo_reslock[test_index]['presResult'])
            y_test = y_test.astype(np.float64)
            train_times = {'start': -0.2, 'stop': 0.8}
            test_times = {'start': -0.2, 'stop': 0.8}

        elif (train_set == 'op1') & (test_set == 'resultlock_pres_i'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            #train_index = trialinfo['operand1'] != trialinfo['presResult']
            X_train = epochs
            y_train = np.array(trialinfo['operand1'])
            y_train = y_train.astype(np.float64)
            test_index = (trialinfo_reslock['presResult'] >= 3) & (trialinfo_reslock['presResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['absdeviant'] != 0)
            X_test = epochs_reslock[test_index]
            y_test = np.array(trialinfo_reslock[test_index]['presResult'])
            y_test = y_test.astype(np.float64)
            train_times = {'start': -0.2, 'stop': 0.8}
            test_times = {'start': -0.2, 'stop': 0.8}

        elif (train_set == 'op1') & (test_set == 'cres'):
            #mode = 'cross-validation' # just to see what happens
            epoch_cres = epochs.copy()
            epoch_cres.crop(1.5, 2.4)
            epoch_cres.times = np.arange(-0.1, 0.8008, 0.008) # This depends on the decimate factor and fsample
            epochs.crop(-0.1, 0.8)
            #train_index = trialinfo['operand1']
            test_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_train = epochs
            y_train = np.array(trialinfo['operand1'])
            y_train = y_train.astype(np.float64)
            X_test = epoch_cres[test_index]
            y_test = np.array(trialinfo[test_index]['corrResult'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': np.min(epochs.times), 'stop': np.max(epochs.times)}
            test_times = {'start': np.min(epoch_cres.times), 'stop': np.max(epoch_cres.times)}

        elif (train_set == 'resultlock_cres') & (test_set == 'cres_-200_800'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            # Set train and test set
            train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            #train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['deviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)

            # Test
            epoch_cres = epochs.copy()
            epoch_cres.crop(-.2, .8)

            test_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_test = epoch_cres[test_index]
            y_test = np.array(trialinfo[test_index]['corrResult'])
            y_test = y_test.astype(np.float64)

            train_times = {'start': -.2, 'stop': .8}
            test_times = train_times
            mode = 'cross-validation'

        elif (train_set == 'resultlock_cres') & (test_set == 'cres_800_1600'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            # Set train and test set
            train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            #train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['deviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)

            # Test
            epoch_cres = epochs.copy()
            epoch_cres.crop(.8, 1.6)
            epoch_cres.times = np.arange(0, 0.8008, 0.008) # This depends on the decimate factor and fsample

            test_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_test = epoch_cres[test_index]
            y_test = np.array(trialinfo[test_index]['corrResult'])
            y_test = y_test.astype(np.float64)

            train_times = {'start': 0, 'stop': .8}
            test_times = train_times
            mode = 'cross-validation'

        elif (train_set == 'resultlock_cres') & (test_set == 'cres_1600_2400'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            # Set train and test set
            train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            #train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['deviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)

            # Test
            epoch_cres = epochs.copy()
            epoch_cres.crop(1.6, 2.4)
            epoch_cres.times = epoch_cres.times = np.arange(0, 0.8008, 0.008) # This depends on the decimate factor and fsample

            test_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_test = epoch_cres[test_index]
            y_test = np.array(trialinfo[test_index]['corrResult'])
            y_test = y_test.astype(np.float64)

            train_times = {'start': 0, 'stop': .8}
            test_times = train_times
            mode = 'cross-validation'

        elif (train_set == 'resultlock_cres') & (test_set == 'cres_2400_3200'):
            fname_calc = dirs['data'] + subject + '_calc_lp30_TLresult.mat'
            epochs_reslock, trialinfo_reslock = fldtrp2mne_calc(fname_calc, 'data', 'calc')
            epochs_reslock.decimate(decimate)
            # Set train and test set
            train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0)
            #train_index = (trialinfo_reslock['corrResult'] >= 3) & (trialinfo_reslock['corrResult'] <= 6) & (trialinfo_reslock['operator'] != 0) & (trialinfo_reslock['deviant'] != 0)
            X_train = epochs_reslock[train_index]
            y_train = np.array(trialinfo_reslock[train_index]['corrResult'])
            y_train = y_train.astype(np.float64)

            # Test
            epoch_cres = epochs.copy()
            epoch_cres.crop(2.4, 3.2)
            epoch_cres.times = epoch_cres.times = np.arange(0, 0.8008, 0.008) # This depends on the decimate factor and fsample

            test_index = (trialinfo['corrResult'] >= 3) & (trialinfo['corrResult'] <= 6) & (trialinfo['operator'] != 0)
            X_test = epoch_cres[test_index]
            y_test = np.array(trialinfo[test_index]['corrResult'])
            y_test = y_test.astype(np.float64)

            train_times = {'start': 0, 'stop': .8}
            test_times = train_times
            mode = 'cross-validation'


        elif (train_set == 'vsa') & (test_set == 'addsub'):
            train_index = info_vsa['congruency'] == 1
            test_index = trialinfo['operator'] != 0
            X_train = epoch_vsa[train_index]
            y_train = np.array(info_vsa[train_index]['cue'])
            # Correct labels for the cue to match add and sub
            y_train[y_train == 1] = -1
            y_train[y_train == 2] = 1
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': -0.1, 'stop': 1.5}
            test_times = {'start': 0.7, 'stop': 3.2}

        elif (train_set == 'vsa_len200ms') & (test_set == 'addsub_len200ms'):
            train_index = info_vsa['congruency'] == 1
            test_index = trialinfo['operator'] != 0
            X_train = epoch_vsa[train_index]
            y_train = np.array(info_vsa[train_index]['cue'])
            # Correct labels for the cue to match add and sub
            y_train[y_train == 1] = -1
            y_train[y_train == 2] = 1
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': -0.1, 'stop': 1.5, 'length': 0.2}
            test_times = {'start': 0.7, 'stop': 3.2, 'length': 0.2}

        elif (train_set == 'addsub') & (test_set == 'vsa'):
            train_index = trialinfo['operator'] != 0
            test_index = info_vsa['congruency'] == 1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epoch_vsa[test_index]
            y_test = np.array(info_vsa[test_index]['cue'])
            # Correct labels for the cue to match add and sub
            y_test[y_test == 1] = -1
            y_test[y_test == 2] = 1
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 3.2}
            test_times = {'start': -0.1, 'stop': 1.5}

        elif (train_set == 'addsub') & (test_set == 'op2'):
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = np.array(trialinfo[train_index]['operand2'])
            # Match labels
            y_test[y_test == 0] = -1
            y_test[y_test == 1] = -1
            y_test[y_test == 2] = 1
            y_test[y_test == 3] = 1
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 1.5}
            test_times = {'start': 1.5, 'stop': 3.1}

        elif (train_set == 'addsub') & (test_set == 'op1'):
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = np.array(trialinfo[train_index]['operand1'])
            # Match labels
            y_test[y_test == 3] = -1
            y_test[y_test == 4] = -1
            y_test[y_test == 5] = 1
            y_test[y_test == 6] = 1
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 1.5}
            test_times = {'start': -.1, 'stop': 0.7}

        elif (train_set == 'op2') & (test_set == 'addsub'):
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            # Match labels
            y_train[y_train == 0] = -1
            y_train[y_train == 1] = -1
            y_train[y_train == 2] = 1
            y_train[y_train == 3] = 1
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = np.array(trialinfo[train_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 1.5, 'stop': 3.1}
            test_times = {'start': 0.7, 'stop': 1.5}

        elif (train_set == 'op1') & (test_set == 'addsub'):
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand1'])
            # Match labels
            y_train[y_train == 3] = -1
            y_train[y_train == 4] = -1
            y_train[y_train == 5] = 1
            y_train[y_train == 6] = 1
            y_train = y_train.astype(np.float64)
            X_test = X_train
            y_test = np.array(trialinfo[train_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': -.1, 'stop': 0.7}
            test_times = {'start': 0.7, 'stop': 1.5}

        # Riemann
        elif (train_set == 'addsub_riemann') & (test_set == 'op2_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[train_index]
            y_test = np.array(trialinfo[train_index]['operand2'])
            # Match labels
            y_test[y_test == 0] = -1
            y_test[y_test == 1] = -1
            y_test[y_test == 2] = 1
            y_test[y_test == 3] = 1
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(.785, 1.485)
            X_test.crop(1.570, 2.270)
            # Update params
            train_times = {'start': .785, 'stop': 1.485}
            test_times = {'start': 1.570, 'stop': 2.270}

        elif (train_set == 'addsub_riemann') & (test_set == 'op1_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[train_index]
            y_test = np.array(trialinfo[train_index]['operand1'])
            # Match labels
            y_test[y_test == 3] = -1
            y_test[y_test == 4] = -1
            y_test[y_test == 5] = 1
            y_test[y_test == 6] = 1
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(.785, 1.485)
            X_test.crop(0, .700)
            # Update params
            train_times = {'start': .785, 'stop': 1.485}
            test_times = {'start': 0, 'stop': .700}

        elif (train_set == 'op2_riemann') & (test_set == 'addsub_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            # Match labels
            y_train[y_train == 0] = -1
            y_train[y_train == 1] = -1
            y_train[y_train == 2] = 1
            y_train[y_train == 3] = 1
            y_train = y_train.astype(np.float64)
            X_test = epochs[train_index]
            y_test = np.array(trialinfo[train_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(1.570, 2.270)
            X_test.crop(.785, 1.485)
            # Update params
            train_times = {'start': 1.570, 'stop': 2.270}
            test_times = {'start': .785, 'stop': 1.485}

        elif (train_set == 'op1_riemann') & (test_set == 'addsub_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand1'])
            # Match labels
            y_train[y_train == 3] = -1
            y_train[y_train == 4] = -1
            y_train[y_train == 5] = 1
            y_train[y_train == 6] = 1
            y_train = y_train.astype(np.float64)
            X_test = epochs[train_index]
            y_test = np.array(trialinfo[train_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(0, .700)
            X_test.crop(.785, 1.485)
            # Update params
            train_times = {'start': 1.570, 'stop': 2.270}
            test_times = {'start': .785, 'stop': 1.485}

        elif (train_set == 'op2_add') & (test_set == 'op2_sub'):
            train_index = trialinfo['operator'] == 1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)

            test_index = trialinfo['operator'] == -1
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operand2'])
            y_test = y_test.astype(np.float64)
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif (train_set == 'op2_sub') & (test_set == 'op2_add'):
            train_index = trialinfo['operator'] == -1
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operand2'])
            y_train = y_train.astype(np.float64)

            test_index = trialinfo['operator'] == 1
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operand2'])
            y_test = y_test.astype(np.float64)
            train_times = {'start': 1.6, 'stop': 3.2}  # 'length': 0.05 defonce memory!
            test_times = train_times

        elif (train_set == 'addsub') & (test_set == 'op2_0'):
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 1.5}
            test_times = {'start': 1.5, 'stop': 3.1}

        elif (train_set == 'addsub') & (test_set == 'op2_1'):
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 1)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 1.5}
            test_times = {'start': 1.5, 'stop': 3.1}

        elif (train_set == 'addsub') & (test_set == 'op2_2'):
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 2)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 1.5}
            test_times = {'start': 1.5, 'stop': 3.1}

        elif (train_set == 'addsub') & (test_set == 'op2_3'):
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 3)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Update params
            train_times = {'start': 0.7, 'stop': 1.5}
            test_times = {'start': 1.5, 'stop': 3.1}

        # Riemann
        elif (train_set == 'addsub_riemann') & (test_set == 'op2_0_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 0)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(.785, 1.485)
            X_test.crop(1.570, 2.270)
            # Update params
            train_times = {'start': .785, 'stop': 1.485}
            test_times = {'start': 1.570, 'stop': 2.270}

        # Riemann
        elif (train_set == 'addsub_riemann') & (test_set == 'op2_1_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 1)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(.785, 1.485)
            X_test.crop(1.570, 2.270)
            # Update params
            train_times = {'start': .785, 'stop': 1.485}
            test_times = {'start': 1.570, 'stop': 2.270}

        # Riemann
        elif (train_set == 'addsub_riemann') & (test_set == 'op2_2_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 2)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(.785, 1.485)
            X_test.crop(1.570, 2.270)
            # Update params
            train_times = {'start': .785, 'stop': 1.485}
            test_times = {'start': 1.570, 'stop': 2.270}

        # Riemann
        elif (train_set == 'addsub_riemann') & (test_set == 'op2_3_riemann'):
            epochs.pick_types(meg='grad')
            train_index = trialinfo['operator'] != 0
            test_index = (trialinfo['operator'] != 0) & (trialinfo['operand2'] == 3)
            X_train = epochs[train_index]
            y_train = np.array(trialinfo[train_index]['operator'])
            y_train = y_train.astype(np.float64)
            X_test = epochs[test_index]
            y_test = np.array(trialinfo[test_index]['operator'])
            y_test = y_test.astype(np.float64)
            # Cropping
            X_train.crop(.785, 1.485)
            X_test.crop(1.570, 2.270)
            # Update params
            train_times = {'start': .785, 'stop': 1.485}
            test_times = {'start': 1.570, 'stop': 2.270}




    print(train_set)
    print(test_set)
    print('done')

    # Check if trialinfo has been defined
    try:
        print(X_train_info)
    except:
        X_train_info = []

    # Organize parameters
    params = {'subject': subject, 'baseline_correction': baselinecorr,
             'train_set': train_set, 'test_set': test_set,
              'train_times': train_times, 'test_times': test_times,
              'times': epochs.times,
              'mode': mode,'baseline': baseline, 'decimate': decimate,
              'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'X_train_info': X_train_info}

    print(params)

    # params = {'subject': subject,
    #          'train_set': train_set, 'test_set': test_set,
    #           'train_times': train_times, 'test_times': test_times,
    #           'times_calc': times_calc, 'times_vsa': times_vsa,
    #           'mode': mode,'baseline': baseline, 'downsampling': downsampling,
    #           'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

    return params