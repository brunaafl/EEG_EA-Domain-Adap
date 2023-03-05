"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>

Baseline script to analyse the EEG Dataset.

"""

import torch
from moabb.datasets import BNCI2014001, PhysionetMI
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline

from pipeline import ClassifierModel, TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
from train import define_clf, init_model
from util import parse_args, set_determinism, set_run_dir

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
    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014001()
        datasets = [dataset]
        n_chans = 22
        input_window_samples = 1001
        rpc = 12
        events = ["right_hand", "left_hand"]
        n_classes = len(events)

    else:
        dataset = PhysionetMI(imagined=True)
        datasets = [dataset]
        n_chans = 64
        input_window_samples = 481
        rpc = 6
        events = ["left_hand", "right_hand"]
        n_classes = len(events)

    paradigm = MotorImagery(events=events, n_classes=len(events))

    model = init_model(n_chans, n_classes, input_window_samples, config=config)
    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, config)

    # Create pipeline
    if args.alignment == 'alignment':
        create_dataset = TransformaParaWindowsDatasetEA(rpc, n_classes)
    else:
        create_dataset = TransformaParaWindowsDataset()

    fit_params = {"epochs": config.train.n_epochs}

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
        hdf5_path=run_dir,
    )

    results = evaluation.process(pipes)
    print(results.head())

    # Save results
    results.to_csv(f"{run_dir}/{experiment_name}_results.csv")

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
