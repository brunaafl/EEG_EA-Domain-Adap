"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""

import torch
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from tqdm import tqdm

from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI
from moabb.utils import set_download_dir

from train import init_model, clf_tuning, define_clf
from util import parse_args, set_determinism, set_run_dir
from dataset import split_runs_EA, delete_trials
from paradigm import MotorImagery_

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import LabelEncoder

"""
For the shared model
"""


def main(args):
    """
    Parameters
    ----------
    args : object
    """
    config = OmegaConf.load(args.config_file)
    # Setting run information
    set_determinism(seed=config.seed)
    # Set download dir
    run_dir, experiment_name = set_run_dir(config, args)
    set_download_dir(config.dataset.path)
    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    # Define paradigm and datasets
    events = ["right_hand", "left_hand"]
    channels = ["Fz", "FC3", "FCz", "FC4", "C5", "FC1", "FC2",
                "C3", "C4", "Cz", "C6", "CPz", "C1", "C2",
                "CP2", "CP1", "CP4", "CP3", "Pz", "P2", "P1", "POz"]

    paradigm = MotorImagery_(events=events, n_classes=len(events), metric='accuracy', channels=channels, resample=250)

    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014001()
    elif args.dataset == 'Cho2017':
        dataset = Cho2017()
    elif args.dataset == 'Lee2019_MI':
        dataset = Lee2019_MI()
    elif args.dataset == 'Schirrmeister2017':
        dataset = Schirrmeister2017()
    elif args.dataset == 'PhysionetMI':
        dataset = PhysionetMI()
        # paradigm = LeftRightImagery_(resample=100.0, metric='accuracy')

    datasets = [dataset]
    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    X_, labels_, meta_ = paradigm.get_data(dataset=dataset, subjects=[1])
    n_chans = X_.shape[1]
    input_window_samples = X_.shape[2]

    model = init_model(n_chans, n_classes, input_window_samples, config=config)
    # Send model to GPU
    if cuda:
        model.cuda()

    clf = clf_tuning(model, config)

    # Epochs array for whole dataset
    X, y, meta = paradigm.get_data(dataset=dataset, return_epochs=False)
    groups = meta.subject.values
    sessions = meta.session.values
    runs = meta.run.values
    n_subjects = len(dataset.subject_list)
    le = LabelEncoder()
    y = le.fit_transform(y)

    if dataset.code == "Schirrmeister2017":
        ea = config.ea.batch
        train_idx = delete_trials(X, y, groups, config.seed, ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]
        sessions = sessions[train_idx]
        runs = runs[train_idx]

    cv = LeaveOneGroupOut()

    param_grid = {
        'optimizer__lr': [0.00125, 0.001, 0.000925, 0.000825, 0.000725],
    }

    params = []
    best_score = []
    for test, train in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-IndivModels"):

        grid = GridSearchCV(
            estimator=clone(clf),
            param_grid=param_grid,
            return_train_score=True,
            scoring='accuracy',
            refit=True,
            verbose=1,
            error_score='raise'
        )

        grid.fit(X[train], y[train])
        params.append(grid.best_params_)
        best_score.append(grid.best_score_)

    print(params)
    print(best_score)

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
