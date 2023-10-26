"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""

import torch
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI
from moabb.utils import set_download_dir

from train import init_model, clf_tuning
from util import parse_args, set_determinism, set_run_dir
from sklearn.base import clone
from paradigm import MotorImagery_

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

    group = meta.subject.values

    X_train, X_test, y_train, y_test, group_train, group_test = \
        train_test_split(X.get_data(), y, group, test_size=config.train.valid_split, random_state=config.seed)

    loo = LeaveOneGroupOut()

    param_grid = {
        'optimizer__lr': [0.00125, 0.001, 0.000825],
    }

    search = GridSearchCV(
        estimator=clone(clf),
        param_grid=param_grid,
        cv=loo,
        return_train_score=True,
        scoring='accuracy',
        refit=True,
        verbose=1,
        error_score='raise'
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    clf = search.fit(X_train, y_train, groups=group_train)

    search_results = pd.DataFrame(clf.cv_results_)

    best_run = search_results[search_results['rank_test_score'] == 1].squeeze()
    print(f"Best hyperparameters were {best_run['params']} which gave a validation "
          f"accuracy of {best_run['mean_test_score'] * 100:.2f}% (training "
          f"accuracy of {best_run['mean_train_score'] * 100:.2f}%).")

    score = search.score(X_test, y_test)
    print(f"Eval accuracy is {score * 100:.2f}%.")

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
