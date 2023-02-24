import logging
from copy import deepcopy
from time import time
from typing import Union

import numpy as np
from mne.epochs import BaseEpochs
from sklearn.metrics import get_scorer
from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

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
            cv = LeaveOneOut()
            # Progressbar at subject level
            for test, train in tqdm(
                cv.split(X, y, groups),
                total=n_subjects,
                desc=f"{dataset.code}-WithinSubject"):

                #aux = np.unique(groups[test])
                #subj_0 = aux[0]
                run_pipes = self.results.not_yet_computed(pipelines, dataset, train[0])

                for name, clf in run_pipes.items():
                    t_start = time()
                    model = deepcopy(clf).fit(X[train], y[train])
                    duration = time() - t_start

                    # for each test subject
                    for subject in np.unique(groups[test]):
                        # Now evaluate
                        ix = groups[test] == subject
                        score = _score(model, X[test[ix]], y[test[ix]], scorer)

                        nchan = (
                            X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                        )
                        res = {
                            "time": duration,
                            "dataset": dataset,
                            "subject": subject,
                            #"session": session,
                            "score": score,
                            "n_samples": len(train),
                            "n_channels": nchan,
                            "pipeline": name,
                        }

                        yield res

    def is_valid(self, dataset):
        return len(dataset.subject_list) > 1