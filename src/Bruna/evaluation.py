import logging
from copy import deepcopy
from time import time
from typing import Union

import numpy as np
from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer
from sklearn.model_selection import LeaveOneGroupOut

from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation
import pandas as pd
import mne

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
                            "dataset": dataset,
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


def add_test_column(dataset, results):
    subj_list = dataset.subject_list
    list_ = []
    for i in subj_list:
        subj_copy = deepcopy(subj_list)
        subj_copy.remove(i)
        list_.append(subj_copy)

    array = np.array(list_)
    test = array.flatten()
    results.insert(4, 'test', test, True)
    return results


def eval_exp2(dataset, paradigm, pipes):
    """

    Cross subject evaluation with different sizes of training set.
    For each test subject, at each assay, we chose

    :param dataset:
    :param paradigm:
    :param pipes:
    :return: results:
             model_list:

    """

    X, y, metadata = paradigm.get_data(dataset, return_epochs=True)
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

    results = list()
    # for each test subject
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CSTrainSize"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            train_idx = runs[train] == 'run_0'
            runs_list = np.unique(runs[train])
            runs_idx = list(range(len(runs_list)))

            # MAYBE it could be interesting to sort the runs instead of use the order
            for r in runs_idx:
                tr = runs[train] == f"run_{r}"
                train_idx = np.logical_or(train_idx, tr)

                t_start = time()
                model = deepcopy(clf).fit(X[train[train_idx]], y[train[train_idx]])
                duration = time() - t_start

                session = 'both'

                # I don't think we need to divide in sessions
                # ix = sessions[test] == session
                score = _score(model, X[test], y[test], scorer)

                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "time": duration,
                    "subject": subject,
                    "n_train_runs": r + 1,
                    "session": session,
                    "score": score,
                    "n_samples": len(train[train_idx]),
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }

                results.append(res)

    results = pd.DataFrame(results)

    return results


def eval_exp4(dataset, paradigm, pipes):
    X, y, metadata = paradigm.get_data(dataset, return_epochs=True)
    # extract metadata
    groups = metadata.subject.values
    sessions = metadata.session.values
    n_subjects = len(dataset.subject_list)

    scorer = get_scorer(paradigm.scoring)

    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    # evaluation
    cv = LeaveOneGroupOut()

    results = []
    model_list = []
    # for each test subject
    for test, train in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-IndividualModels"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            t_start = time()
            model = deepcopy(clf).fit(X[train], y[train])
            duration = time() - t_start
            model_list.append(model)

            # for each test subject
            for subj in np.unique(groups[test]):
                # Now evaluate
                ix = groups[test] == subj
                score = _score(model, X[test[ix]], y[test[ix]], scorer)
                session = 'both'
                nchan = (
                    X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                )
                res = {
                    "time": duration,
                    "subject": groups[train[0]],
                    "test": subj,
                    "session": session,
                    "score": score,
                    "n_samples": len(train),
                    "n_channels": nchan,
                    "dataset": dataset.code,
                    "pipeline": name,
                }

                results.append(res)

    results = pd.DataFrame(results)

    return results, model_list
