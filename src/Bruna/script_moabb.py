"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>

Baseline script to analyse the EEG Dataset.

"""
import os.path as osp

import matplotlib.pyplot as plt
import mne
import seaborn as sns
import torch
import copy

from braindecode import EEGClassifier
from braindecode.datasets import create_from_X_y
from braindecode.models import ShallowFBCSPNet, EEGNetv4
from braindecode.util import set_random_seeds

from moabb.datasets import BNCI2014001, PhysionetMI
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.utils import set_download_dir

from sklearn.pipeline import Pipeline

from train import init_model, define_clf
from pipeline import TransformaParaWindowsDataset, ClassifierModel
from util import parse_args

"""
For the joint model
"""


def main(dataset_type='BNCI2014001'):
    """

    Parameters
    ----------
    args : object
    :param dataset_type:
    """

    # Set download dir
    set_download_dir(osp.join(osp.expanduser("~"), "mne_data"))

    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    device = "cuda" if cuda else "cpu"
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20200220  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    # Define paradigm and datasets
    paradigm = LeftRightImagery()
    n_classes = 2

    if dataset_type == 'BNCI2014001':
        # n_classes = 4
        dataset = BNCI2014001()
        datasets = [dataset]
        n_chans = 22
        input_window_samples = 1001

    else:
        dataset = PhysionetMI(imagined=True)
        datasets = [dataset]
        n_chans = 64
        input_window_samples = 481

    model = init_model(n_chans, n_classes, input_window_samples)
    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, device)

    # Create pipeline
    create_dataset = TransformaParaWindowsDataset()
    fit_params = {"epochs": 200}
    brain_clf = ClassifierModel(clf, fit_params)
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", brain_clf)])
    pipes = {"EEGNetv4": pipe}

    # Define evaluation and train
    overwrite = True  # set to True if we want to overwrite cached results
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        suffix="experiment_1",
        overwrite=overwrite,
        return_epochs=True,
    )

    results = evaluation.process(pipes)
    print(results.head())

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
