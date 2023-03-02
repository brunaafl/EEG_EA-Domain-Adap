"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>

Baseline script to analyse the EEG Dataset.

"""
import os.path as osp

import torch
import copy

from braindecode.util import set_random_seeds

from moabb.datasets import BNCI2014001, PhysionetMI
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation
from moabb.paradigms import LeftRightImagery
from moabb.utils import set_download_dir

from sklearn.pipeline import Pipeline

from train import init_model, define_clf
from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA, ClassifierModel
from util import parse_args
from evaluation import CrossCrossSubjectEvaluation, add_test_column


"""
For the joint model
"""


def main(dataset_type='BNCI2014001', alignment=False):
    """

    Parameters
    ----------
    args : object
    :param alignment:
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
        rpc = 12

    else:
        dataset = PhysionetMI(imagined=True)
        datasets = [dataset]
        n_chans = 64
        input_window_samples = 481
        rpc = 6

    model = init_model(n_chans, n_classes, input_window_samples)
    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, device)

    # Create pipeline
    if alignment:
        create_dataset = TransformaParaWindowsDatasetEA(rpc, n_classes)
    else:
        create_dataset = TransformaParaWindowsDataset()
    fit_params = {"epochs": 200}
    brain_clf = ClassifierModel(clf, fit_params)
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", brain_clf)])
    pipes = {"EEGNetv4": pipe}

    # Define evaluation and train
    overwrite = True  # set to True if we want to overwrite cached results
    # The ONLY difference is going to be on the evaluation
    evaluation = CrossCrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        suffix="experiment_4",
        overwrite=overwrite,
        return_epochs=True,
    )

    results = evaluation.process(pipes)
    results = add_test_column(datasets[0], results)
    print(results.head())

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
