# GAT functions
# Calc MEG
# Pinheiro-Chagas 2017

# Libraries
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score


def Classification(X_train, y_train, X_test, y_test, scorer, predict_mode, params):
    " Multiclass classification within or across conditions "

    ### Learning machinery

    # Model
    model = svm.SVC(class_weight='balanced')

    # Cross-validation scheme
    cv = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)

    # Scaler
    scaler = StandardScaler()

    # Pipeline
    clf = make_pipeline(scaler, model)

    # Define scorer
    if scorer is 'scorer_auc':
        scorer = 'roc_auc'
    elif scorer is 'accuracy':
        scorer = None
    else:
        print('using accuracy as the scorer')

    # Learning and scoring
    time_gen = GeneralizingEstimator(clf, n_jobs=2, scoring=scorer)
    scores = cross_val_multiscore(time_gen, X_train, y_train, cv=cv, n_jobs=2)

    return scores


def LogisticRegression(X_train, y_train, X_test, y_test, scorer, predict_mode, params):
    " Logistic Regression within or across conditions "

    # Model
    model = linear_model.LogisticRegression(class_weight='balanced')

    # Cross-validation scheme
    cv = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)

    # Scaler
    scaler = StandardScaler()

    # Pipeline
    clf = make_pipeline(scaler, model)

    # Define scorer
    if scorer is 'scorer_auc':
        scorer = 'roc_auc'
    elif scorer is 'accuracy':
        scorer = None
    else:
        print('using accuracy as the scorer')

    # Learning and scoring
    time_gen = GeneralizingEstimator(clf, n_jobs=2, scoring=scorer)
    scores = cross_val_multiscore(time_gen, X_train, y_train, cv=cv, n_jobs=2)

    return scores


def NeuralNet(X_train, y_train, X_test, y_test, scorer, predict_mode, params):
    " Neural Network estimator "

    # Model
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=1)

    # Cross-validation scheme
    cv = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)

    # Scaler
    scaler = StandardScaler()

    # Pipeline
    clf = make_pipeline(scaler, model)

    # Define scorer
    if scorer is 'scorer_auc':
        scorer = 'roc_auc'
    elif scorer is 'accuracy':
        scorer = None
    else:
        print('using accuracy as the scorer')

    # Learning and scoring
    time_gen = GeneralizingEstimator(clf, n_jobs=2, scoring=scorer)
    scores = cross_val_multiscore(time_gen, X_train, y_train, cv=cv, n_jobs=2)

    return scores