"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""

import torch
import pandas as pd
import numpy as np

from omegaconf import OmegaConf
from copy import deepcopy
from time import time
from tqdm import tqdm
from mne.epochs import BaseEpochs

from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder


import moabb.analysis.plotting as moabb_plt
from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI
from moabb.paradigms import MotorImagery, LeftRightImagery
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
import matplotlib.pyplot as plt

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
from train import define_clf, init_model
from util import parse_args, set_determinism, set_run_dir
from sklearn.base import clone

"""
For the joint model
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

    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    # Define paradigm and datasets
    events = ["right_hand", "left_hand"]

    paradigm = MotorImagery(events=events, n_classes=len(events))

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
        paradigm = LeftRightImagery()

    datasets = [dataset]
    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    n_chans = X.shape[1]
    input_window_samples = X.shape[2]
    rpc = len(meta['session'].unique())*len(meta['run'].unique())

    model = init_model(n_chans, n_classes, input_window_samples, config=config)
    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, config)

    # Create pipeline
    create_dataset_with_align = TransformaParaWindowsDatasetEA(rpc, n_classes)
    create_dataset = TransformaParaWindowsDataset()

    pipes = {}

    pipe_with_align = Pipeline([("Braindecode_dataset", create_dataset_with_align),
                                ("Net", clone(clf))])
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", clone(clf))])

    pipes["EEGNetv4_EA"] = pipe_with_align
    pipes["EEGNetv4_Without_EA"] = pipe

    # Define evaluation and train
    overwrite = True  # set to True if we want to overwrite cached results
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

    dataset_res = list()
    # for each test subject
    for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject"):

        subject = groups[test[0]]

        # iterate over each pipeline
        for name, clf in pipes.items():

            train_idx = runs[train] == 'run_0'
            runs_list = np.unique(runs[train])
            runs_idx = list(range(len(runs_list)))

            # MAYBE it could be interesting to sort the runs to ass instead of use the order
            for r in runs_idx:
                # if sorted: k=0
                # while k < len(runs_idx)

                # If we want sorted:
                # if r =! 0:
                #  r = choice(runs_idx)
                #  runs_idx.remove(r)
                #

                tr = runs[train] == f"run_{r}"
                train_idx = np.logical_or(train_idx, tr)

                t_start = time()
                model = deepcopy(clf).fit(X[train[train_idx]], y[train[train_idx]])
                duration = time() - t_start

                session = 'both'

                # I don't think we need to divide in sessions
                # ix = sessions[test] == session
                score = _score(model, X[test], y[test], scorer)
                print(score)

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

                dataset_res.append(res)

    results = pd.DataFrame(dataset_res)

    # results = evaluation.process(pipes)
    print(results.head())

    # Save results
    results.to_csv(f"{run_dir}/{experiment_name}_results.csv")

    fig, color_dict = moabb_plt.score_plot(results)
    fig.savefig(f"{run_dir}/score_plot_models.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig = moabb_plt.paired_plot(results, "EEGNetv4_EA", "EEGNetv4_Without_EA")
    fig.savefig(f"{run_dir}/paired_score_plot_models.pdf", format='pdf', dpi=300, bbox_inches='tight')

    stats = compute_dataset_statistics(results)
    P, T = find_significant_differences(stats)

    fig = moabb_plt.meta_analysis_plot(stats, "EEGNetv4_EA", "EEGNetv4_Without_EA")
    fig.savefig(f"{run_dir}/meta_analysis_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig = moabb_plt.summary_plot(P, T)
    fig.savefig(f"{run_dir}/meta_analysis_summary_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)