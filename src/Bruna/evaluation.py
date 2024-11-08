import logging
from typing import Union

from moabb.evaluations.base import BaseEvaluation

from mne.decoding import Scaler

import mne
import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from time import time

from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from mlxtend.classifier import EnsembleVoteClassifier

from tqdm import tqdm

from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit

from pipeline import TransformaParaWindowsDatasetEA, TransformaParaWindowsDataset
from dataset import split_runs_EA, delete_trials
from alignment import euclidean_alignment
from riemann import riemannian_alignment, resting_alignment
from train import define_clf

mne.set_log_level(False)
log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]


class CrossCrossSubjectEvaluation(BaseEvaluation):
    """
    Temporary name!
    I want to create the evaluation that I did in the experiment 4. For this, I created one
    classifier per subject, and evaluated using the other ones.

    Parameters
    ----------
    paradigm : Paradigm instance
        The paradigm to use.
    datasets : List of Dataset instance
        The list of dataset to run the evaluation. If none, the list of
        compatible dataset will be retrieved from the paradigm instance.
    random_state: int, RandomState instance, default=None
        If not None, can guarantee same seed for shuffling examples.
    n_jobs: int, default=1
        Number of jobs for fitting of pipeline.
    overwrite: bool, default=False
        If true, overwrite the results.
    error_score: "raise" or numeric, default="raise"
        Value to assign to the score if an error occurs in estimator fitting. If set to
        'raise', the error is raised.
    suffix: str
        Suffix for the results file.
    hdf5_path: str
        Specific path for storing the results.
    additional_columns: None
        Adding information to results.
    return_epochs: bool, default=False
        use MNE epoch to train pipelines.
    mne_labels: bool, default=False
        if returning MNE epoch, use original dataset label if True
    """

    def evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        # this is a bit akward, but we need to check if at least one pipe
        # have to be run before loading the data. If at least one pipeline
        # need to be run, we have to load all the data.
        # we might need a better granularity, if we query the DB
        run_pipes = {}
        for subject in dataset.subject_list:
            run_pipes.update(self.results.not_yet_computed(pipelines, dataset, subject))
        if len(run_pipes) != 0:

            # get the data
            X, y, metadata = self.paradigm.get_data(dataset,
                                                    return_epochs=self.return_epochs)

            # encode labels
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)

            # extract metadata
            groups = metadata.subject.values
            sessions = metadata.session.values
            n_subjects = len(dataset.subject_list)

            scorer = get_scorer(self.paradigm.scoring)

            # perform leave one subject out CV
            cv = LeaveOneGroupOut()
            # Progressbar at subject level
            model_list = []
            for test, train in tqdm(
                    cv.split(X, y, groups),
                    total=n_subjects,
                    desc=f"{dataset.code}-CrossCrossSubject"):

                # aux = np.unique(groups[test])
                # subj_0 = aux[0]
                run_pipes = self.results.not_yet_computed(pipelines, dataset, groups[train[0]])

                for name, clf in run_pipes.items():
                    t_start = time()
                    model = deepcopy(clf).fit(X[train], y[train])
                    duration = time() - t_start
                    model_list.append(model)

                    # for each test subject
                    for subject in np.unique(groups[test]):
                        # Now evaluate
                        ix = groups[test] == subject
                        score = _score(model, X[test[ix]], y[test[ix]], scorer)
                        session = 'both'
                        nchan = (
                            X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                        )
                        res = {
                            "time": duration,
                            "dataset": dataset.code,
                            "subject": groups[train[0]],
                            "test": subject,
                            "session": session,
                            "score": score,
                            "n_samples": len(train),
                            "n_channels": nchan,
                            "pipeline": name,
                        }

                        yield res

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1


def shared_model(dataset, paradigm, pipes, run_dir, config, align=None):
    """

    Create one model per subject and the with the others

    :param run_dir:
    :param dataset : moabb.datasets
    :param paradigm : moabb.paradigms
    :param pipes : Pipeline
    :return: results : df
             model_list: list of clf

    """
    X, y, metadata = paradigm.get_data(dataset=dataset)  # Removing return_epochs = True
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    # Delete some trials
    if dataset.code == "Schirrmeister2017":
        len_ea = config.ea.batch
        train_idx = delete_trials(X, y, groups, config.seed, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]
        sessions = sessions[train_idx]
        runs = runs[train_idx]

    results = []
    # for each test subject
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-SharedModels"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            cvclf = deepcopy(clf)
            t_start = time()
            model = cvclf.fit(X[train], y[train])
            duration = time() - t_start

            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp1.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp1.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp1.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp1.pkl"),
            )

            nchan = (
                X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
            )

            for session in np.unique(sessions[test]):

                # Select runs used for the EA test
                # test_runs we are going to use for test
                # aux_run we are going to use for the EA
                test_runs, aux_run = select_run(runs, sessions, test, dataset.code, session, groups)
                len_run = sum(aux_run * 1)

                # We exclude aux EA trials so the results are comparable
                ix = sessions[test] == session

                # Test part using just trials from this session
                # And without the aux EA trials
                test_idx = np.logical_and(test_runs, ix)
                Test = X[test[test_idx]]
                y_t = y[test[test_idx]]
                model['Braindecode_dataset'].y = y_t
                score = _score(model, Test, y_t, scorer)

                res = {
                    "time": duration,
                    "dataset": dataset.code,
                    "subject": subject,
                    "session": session,
                    "score": score,
                    "type": "Offline",
                    "ft": "Without",
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                    "exp": "shared"
                }

                results.append(res)

                # If we are analyzing with EA
                if align is not None:

                    # Then, use the calibration run
                    Aux_trials = X[test[aux_run]]
                    if align == 'alignment':
                        _, r_op = euclidean_alignment(Aux_trials)
                    elif align == 'r-alignment':
                        _, r_op = riemannian_alignment(Aux_trials)
                    elif align == 'rest-alignment':
                        tbreak = model['Braindecode_dataset'].tbreak
                        _, r_op = resting_alignment(Aux_trials, tbreak)
                        Test = Test[:, :, :tbreak]
                    # Use ref matrix to align test data
                    X_t = np.matmul(r_op, Test)
                    # Compute score
                    score_EA = _score(model["Net"], X_t, y_t, scorer)

                # Else, no changes for ea
                else:
                    score_EA = score

                res = {
                    "time": duration,
                    "dataset": dataset.code,
                    "subject": subject,
                    "session": session,
                    "score": score_EA,
                    "type": "Online",
                    "ft": "Without",
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "pipeline": name,
                    "exp": "1run"
                }
                results.append(res)

    results = pd.DataFrame(results)
    return results


def ftdata(runs, sessions, session, train, groups, dataset, ea=24):
    """
    Select the run that is going to be used as auxiliar

    :param groups:
    :param session:
    :param ea:
    :param train:
    :param dataset:
    :param runs: array indicating each trial's run
    :param sessions: array indicating each trial's session

    :return:
        :param test_runs: boolean array with True in the index of test trials (and False elsewhere)
        :param aux_run : boolean array with True in the index of aux trials (and False elsewhere)
    """

    runs_ = np.unique(runs[train])

    if dataset == '001-2014':
        # Select the first run
        r = runs[train] == runs_[0]
        s = sessions[train] == session

        aux_run = np.logical_and(r, s)

    elif dataset == 'Schirrmeister2017':
        trials = []
        for subj in np.unique(groups[train]):
            g = groups[train] == subj
            len_subj = sum(g)
            first_trials = np.ones(len_subj, dtype=bool)
            first_trials[int(ea):] = 0
            trials.append(first_trials)
        aux_run = np.concatenate(trials)

    # train_idx = np.concatenate((train[aux_run], aux_test))

    return aux_run


def select_run(runs, sessions, test, dataset, session, groups, ea=24):
    """
    Select the run that is going to be used as auxiliar

    :param ea:
    :param dataset:
    :param runs: array indicating each trial's run
    :param sessions: array indicating each trial's session
    :param test: array with the index of test trials
    :param session: string (name of session)

    :return:
        :param test_runs: boolean array with True in the index of test trials (and False elsewhere)
        :param aux_run : boolean array with True in the index of aux trials (and False elsewhere)
    """

    runs_ = np.unique(runs[test])

    if dataset == '001-2014':

        # Select the first run from given session
        r = runs[test] == runs_[0]
        s = sessions[test] == session

        aux_run = np.logical_and(r, s)

    elif dataset == 'Schirrmeister2017':

        trials = []
        for subj in np.unique(groups[test]):
            g = groups[test] == subj
            len_subj = sum(g)
            first_trials = np.ones(len_subj, dtype=bool)
            first_trials[int(ea):] = 0
            trials.append(first_trials)
        r = np.concatenate(trials)
        s = sessions[test] == session

        aux_run = np.logical_and(r, s)

        # r = runs[test] == 'train'
        # r[int(ea):] = False
        # s = sessions[test] == session

    # Select the opposit for the test
    test_runs = np.invert(aux_run)

    return test_runs, aux_run


def online_shared(dataset, paradigm, pipes, nn_model, run_dir, config):
    """

    Create one model per subject and the with the others

    :param nn_model:
    :param run_dir:
    :param dataset : moabb.datasets
    :param paradigm : moabb.paradigms
    :param pipes : Pipeline
    :return: results : df
             model_list: list of clf

    Parameters
    ----------
    nn_model

    """
    X, y, metadata = paradigm.get_data(dataset)  # Removing return_epochs=True
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    nchan = (
        X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
    )

    # Delete some trials
    if dataset.code == "Schirrmeister2017":
        len_ea = config.ea.batch
        train_idx = delete_trials(X, y, groups, config.seed, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]
        sessions = sessions[train_idx]
        runs = runs[train_idx]

    results = []
    # for each test subject
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-SharedModels"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            ftclf = create_clf_ft(nn_model, config)
            ftclf.initialize()

            # Initialize with the saved parameters
            ftclf.load_params(
                f_params=str(run_dir / f"final_model_params_{subject}_exp1.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_exp1.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_exp1.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_exp1.pkl"), )

            # Freeze some layers
            freeze(ftclf, config)

            # Now test
            for session in np.unique(sessions[test]):

                test_runs, aux_run = select_run(runs, sessions, test, dataset.code, session, groups)
                len_run = sum(aux_run * 1)

                aux_test = test[aux_run]

                # Compute train data
                run_train = ftdata(runs, sessions, session, train, groups, dataset.code)
                train_idx = np.concatenate((train[run_train], aux_test))

                X_train = X[train_idx]
                y_train = y[train_idx]

                ix = sessions[test] == session
                test_idx = np.logical_and(test_runs, ix)
                Test = X[test[test_idx]]
                y_t = y[test[test_idx]]

                if type(pipes[name][0]) == type(TransformaParaWindowsDatasetEA(len_run=len_run)):

                    X_train_ = split_runs_EA(X_train, len_run)
                    t_start = time()
                    ftmodel = ftclf.fit(X_train_, y_train)
                    duration = time() - t_start

                    Aux_trials = X[aux_test]
                    _, r_op = euclidean_alignment(Aux_trials)
                    # Use ref matrix to align test data
                    X_t = np.matmul(r_op, Test)
                else:

                    t_start = time()
                    ftmodel = ftclf.fit(X_train, y_train)
                    duration = time() - t_start

                    X_t = Test

                # Predict on the test set
                score = _score(ftmodel, X_t, y_t, scorer)

                res = {
                    "time": duration,
                    "dataset": dataset.code,
                    "subject": subject,
                    "session": session,
                    "score": score,
                    "type": "Online",
                    "ft": "With",
                    "n_samples": len(y_train),
                    "n_channels": nchan,
                    "pipeline": name,
                    "exp": "fine-tuning"
                }
                results.append(res)

    results = pd.DataFrame(results)
    return results


def freeze(model, config):
    if config.model.type == 'Deep4Net':
        model.module_.conv_time.weight.requires_grad = False
        model.module_.conv_spat.weight.requires_grad = False
        model.module_.bnorm.weight.requires_grad = False
        model.module_.conv_2.weight.requires_grad = False
        model.module_.bnorm_2.weight.requires_grad = False
        model.module_.conv_3.weight.requires_grad = False
        model.module_.bnorm_3.weight.requires_grad = False
        model.module_.conv_4.weight.requires_grad = False
        model.module_.bnorm_4.weight.requires_grad = False
        model.module_.conv_classifier.weight.requires_grad = False

    elif config.model.type == 'EEGNetv4':

        model.module_.conv_temporal.weight.requires_grad = False
        model.module_.bnorm_temporal.weight.requires_grad = False
        model.module_.conv_spatial.weight.requires_grad = False
        model.module_.bnorm_1.weight.requires_grad = False
        model.module_.conv_separable_depth.weight.requires_grad = False
        model.module_.conv_separable_point.weight.requires_grad = False
        model.module_.bnorm_2.weight.requires_grad = False

    else:
        model.module_.conv_time.weight.requires_grad = False
        model.module_.conv_spat.weight.requires_grad = False
        model.module_.bnorm.weight.requires_grad = False
        model.module_.conv_classifier.weight.requires_grad = False

    return model


def individual_models(dataset, paradigm, pipes, run_dir, config):
    """

    Create one model per subject and the with the others

    :param run_dir:
    :param dataset : moabb.datasets
    :param paradigm : moabb.paradigms
    :param pipes : Pipeline
    :return: results : df
             model_list: list of clf

    """
    X, y, metadata = paradigm.get_data(dataset)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    nchan = (
        X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
    )

    # Delete some trials
    if dataset.code == "Schirrmeister2017":
        len_ea = config.ea.batch
        train_idx = delete_trials(X, y, groups, config.seed, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]
        sessions = sessions[train_idx]
        runs = runs[train_idx]

    results = []
    model_list = []

    # for each train subject
    for test, train in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-IndividualModels"):

        subject = groups[train[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            cvclf = deepcopy(clf)
            t_start = time()
            # Fit with data from subject "train"
            model = cvclf.fit(X[train], y[train])
            duration = time() - t_start
            model_list.append([cvclf, model])

            # Test with the same data used as train
            score = _score(model, X[train], y[train], scorer)

            session = 'both'

            res = {
                "time": duration,
                "dataset": dataset.code,
                "subject": subject,
                "test": subject,
                "session": session,
                "score": score,
                "type": "Offline",
                "ft": "Without",
                "n_samples": len(train),
                "n_channels": nchan,
                "pipeline": name,
                "exp": "indiv"
            }

            results.append(res)

            cvclf['Net'].save_params(
                f_params=str(run_dir / f"final_model_params_{subject}_indiv.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_indiv.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_indiv.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_indiv.pkl"),
            )

            # Test subject
            for subj in np.unique(groups[test]):
                for session in np.unique(sessions[test]):

                    # Take data from test subject subj and session
                    test_subj = groups[test] == subj
                    ix = sessions[test] == session

                    # Select runs used for the EA test
                    # test_runs we are going to use for test
                    # aux_run we are going to use for the EA
                    test_runs, aux_run = select_run(runs, sessions, test, dataset.code,
                                                    session, groups, ea=config.ea.batch)

                    # Select just the required part
                    aux_idx = np.logical_and(aux_run, test_subj)
                    len_run = sum(aux_idx * 1)

                    # Select just the required part
                    test_idx = np.logical_and(test_runs, test_subj)
                    #  Select required session
                    test_idx = np.logical_and(test_idx, ix)
                    Test = X[test[test_idx]]
                    y_t = y[test[test_idx]]
                    model['Braindecode_dataset'].y = y_t

                    score = _score(model, Test, y_t, scorer)

                    res = {
                        "time": duration,
                        "dataset": dataset.code,
                        "subject": subject,
                        "test": subj,
                        "session": session,
                        "score": score,
                        "type": "Offline",
                        "ft": "Without",
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                        "exp": "indiv"
                    }

                    results.append(res)

                    # If we are analyzing with EA
                    if type(pipes[name][0]) == type(TransformaParaWindowsDatasetEA(len_run=len_run)):

                        # Then, test with one run for ft
                        Aux_trials = X[test[aux_idx]]

                        _, r_op = euclidean_alignment(Aux_trials)
                        # Use ref matrix to align test data
                        X_t = np.matmul(r_op, Test)
                        # Compute score
                        score_EA = _score(model["Net"], X_t, y_t, scorer)

                    # Else, no changes for zero shot or ea
                    else:
                        score_EA = score

                    res = {
                        "time": duration,
                        "dataset": dataset.code,
                        "subject": subject,
                        "test": subj,
                        "session": session,
                        "score": score_EA,
                        "type": "Online",
                        "ft": "Without",
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                        "exp": "indiv_1run"
                    }
                    results.append(res)

    results = pd.DataFrame(results)

    return results, model_list


def online_indiv(dataset, paradigm, pipes, nn_model, run_dir, config):
    """

    Create one model per subject and the with the others

    :param nn_model:
    :param run_dir:
    :param dataset : moabb.datasets
    :param paradigm : moabb.paradigms
    :param pipes : Pipeline
    :return: results : df
             model_list: list of clf

    Parameters
    ----------
    nn_model

    """
    X, y, metadata = paradigm.get_data(dataset)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    nchan = (
        X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
    )

    # Delete some trials
    if dataset.code == "Schirrmeister2017":
        len_ea = config.ea.batch
        train_idx = delete_trials(X, y, groups, config.seed, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]
        sessions = sessions[train_idx]
        runs = runs[train_idx]

    results = []
    # for each test subject
    for test, train in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-IndividualModels"):

        subject = groups[train[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            ftclf = create_clf_ft(nn_model, config)
            ftclf.initialize()

            # Initialize with the saved parameters
            ftclf.load_params(
                f_params=str(run_dir / f"final_model_params_{subject}_indiv.pkl"),
                f_history=str(run_dir / f"final_model_history_{subject}_indiv.json"),
                f_criterion=str(run_dir / f"final_model_criterion_{subject}_indiv.pkl"),
                f_optimizer=str(run_dir / f"final_model_optimizer_{subject}_indiv.pkl"),
            )

            # Freeze some layers
            freeze(ftclf, config)

            # Now test
            # Keep this division?
            for subj in np.unique(groups[test]):
                for session in np.unique(sessions[test]):

                    # Take data from test subject subj and session
                    test_subj = groups[test] == subj
                    ix = sessions[test] == session

                    # Select runs used for the EA test
                    # test_runs we are going to use for test
                    # aux_run we are going to use for the EA
                    test_runs, aux_run = select_run(runs, sessions, test, dataset.code,
                                                    session, groups, ea=config.ea.batch)
                    # Select just the required part
                    aux_idx = np.logical_and(aux_run, test_subj)
                    len_run = sum(aux_idx * 1)

                    aux_test = test[aux_idx]

                    # Compute train data
                    # run_train = ftdata(runs, sessions, session, train, groups, dataset.code)
                    # train_idx = np.concatenate((train[run_train], aux_test))
                    # X_train = X[train_idx]
                    # y_train = y[train_idx]

                    train_idx = np.concatenate((train, aux_test))
                    X_train = X[train_idx]
                    y_train = y[train_idx]

                    # Select just the required part
                    test_idx = np.logical_and(test_runs, test_subj)
                    #  Select required session
                    test_idx = np.logical_and(test_idx, ix)
                    Test = X[test[test_idx]]
                    y_t = y[test[test_idx]]

                    if type(pipes[name][0]) == type(TransformaParaWindowsDatasetEA(len_run=len_run)):

                        X_train_ = split_runs_EA(X_train, len_run)
                        t_start = time()
                        ftmodel = ftclf.fit(X_train_, y_train)
                        duration = time() - t_start

                        Aux_trials = X[aux_test]
                        _, r_op = euclidean_alignment(Aux_trials)
                        # Use ref matrix to align test data
                        X_t = np.matmul(r_op, Test)
                    else:

                        t_start = time()
                        ftmodel = ftclf.fit(X_train, y_train)
                        duration = time() - t_start

                        X_t = Test

                    # Predict on the test set
                    score = _score(ftmodel, X_t, y_t, scorer)

                    res = {
                        "time": duration,
                        "dataset": dataset.code,
                        "subject": subject,
                        "test": subj,
                        "session": session,
                        "score": score,
                        "type": "Online",
                        "ft": "With",
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                        "exp": "indiv_ft"
                    }

                    results.append(res)

    results = pd.DataFrame(results)
    return results


def create_clf_ft(model, config):
    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    device = "cuda" if cuda else "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True

    weight_decay = config.ft.weight_decay
    batch_size = config.ft.batch_size
    lr = config.ft.lr
    patience = config.ft.patience

    lrscheduler = LRScheduler(policy='CosineAnnealingLR', T_max=config.train.n_epochs - 1, eta_min=0)

    ftclf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=config.ft.n_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
                   lrscheduler,
                   EpochScoring(scoring='accuracy', on_train=True,
                                name='train_acc', lower_is_better=False),
                   EpochScoring(scoring='accuracy', on_train=False,
                                name='valid_acc', lower_is_better=False)],
        device=device,
    )

    ftclf.initialize()
    return ftclf


def select_weights(X_test, y_test, models, n=5, exp=True):
    scores = []
    for model in models:
        y_pr = model.predict(X_test)
        score = accuracy_score(y_test, y_pr)
        scores.append(score)

    scores = np.array(scores)
    scores_idx = np.argsort(scores)[::-1][:n]
    w = scores[scores_idx]

    if exp:
        w = np.exp(w)
        w = w / sum(w)

    else:
        w = w / sum(w)

    return w, scores_idx


def divide(list_2d):
    length = len(list_2d)

    l1 = []
    l2 = []

    for i in range(length):
        l1.append(list_2d[i][0])
        l2.append(list_2d[i][1])

    return l1, l2


def ensemble_simple_load(dataset, paradigm, run_dir, config, model, ea=None):
    X, y, metadata = paradigm.get_data(dataset)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    runs = metadata.run.values
    n_subjects = len(dataset.subject_list)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    nchan = (
        X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
    )

    # Delete some trials
    if dataset.code == "Schirrmeister2017":
        len_ea = config.ea.batch
        train_idx = delete_trials(X, y, groups, config.seed, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]
        sessions = sessions[train_idx]
        runs = runs[train_idx]

    results = []
    model_list = []
    idx_list = []
    # First, load all classifiers

    for s in np.unique(groups):
        clf = define_clf(deepcopy(model), config)
        clf.initialize()

        # Initialize with the saved parameters
        clf.load_params(
            f_params=str(run_dir / f"final_model_params_{s}_indiv.pkl"),
            f_history=str(run_dir / f"final_model_history_{s}_indiv.json"),
            f_criterion=str(run_dir / f"final_model_criterion_{s}_indiv.pkl"),
            f_optimizer=str(run_dir / f"final_model_optimizer_{s}_indiv.pkl"),
        )

        idx_list.append(s)
        model_list.append(clf)

    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-EnsembleModels"):

        # Select the test subject
        subject = groups[test[0]]

        for session in np.unique(sessions):

            # Select session
            ix = sessions[test] == session

            # Select auxiliar trials
            test_runs, aux_run = select_run(runs, sessions, test, dataset.code,
                                            session, groups, ea=config.ea.batch)
            clfs = model_list.copy()
            subj_idx = idx_list.copy()

            itest = subj_idx.index(subject)
            subj_idx.pop(itest)
            clfs.pop(itest)
            n = config.ensemble.n_clf

            # Use this part of the data to select the best classifiers
            X_train = X[test[aux_run]]
            y_train = y[test[aux_run]]

            if ea is not None:
                len_run = ea
                X_train = split_runs_EA(X_train, len_run)

            # X_train = Scaler(X_train)

            w, idx = select_weights(X_train, y_train, clfs, n=n)

            clfs = [clfs[i] for i in idx]
            w = w.tolist()

            eclf = EnsembleVoteClassifier(clfs=clfs, weights=w,
                                          voting=config.ensemble.voting, fit_base_estimators=False)

            create_dataset = TransformaParaWindowsDataset()

            eclf_pipe = Pipeline(
                [("Braindecode_dataset", create_dataset), ("Ensemble", eclf)])  # ('normalize', Scaler()),

            t_start = time()
            emodel = eclf_pipe.fit(X_train, y_train)
            duration = time() - t_start

            # Now, evaluation
            # Simulated online
            # Select required session
            test_idx = np.logical_and(test_runs, ix)
            Test = X[test[test_idx]]
            y_t = y[test[test_idx]]
            create_dataset.y = y_t

            if ea is not None:
                # Then, test with one run for ft
                Aux_trials = X[test[aux_run]]
                _, r_op = euclidean_alignment(Aux_trials)
                # Use ref matrix to align test data
                Test = np.matmul(r_op, Test)

            # Compute score
            y_pr = emodel.predict(Test)
            score = accuracy_score(y_t, y_pr)

            res = {
                "time": duration,
                "dataset": dataset.code,
                "test": subject,
                "session": session,
                "score": score,
                "type": "Online",
                "ft": "Without",
                "n_samples": len(train),
                "n_channels": nchan,
                "exp": f"online_{n}"
            }

            results.append(res)
    results = pd.DataFrame(results)
    return results
