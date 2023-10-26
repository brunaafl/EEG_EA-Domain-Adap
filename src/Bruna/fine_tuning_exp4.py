"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""

import torch
import numpy as np

from pathlib import Path

from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline

from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI
from moabb.utils import set_download_dir

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
from evaluation import individual_models, online_indiv
from train import define_clf, init_model
from util import parse_args, set_determinism, set_run_dir
from sklearn.base import clone
from paradigm import MotorImagery_

"""
For the imdividual model
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

    paradigm = MotorImagery_(events=events, n_classes=len(events),metric='accuracy', channels=channels, resample=250)

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
        #paradigm = LeftRightImagery_(resample=100.0, metric='accuracy')

    datasets = [dataset]
    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    n_chans = X.shape[1]
    input_window_samples = X.shape[2]
    runs = meta.run.values
    sessions = meta.session.values
    one_session = sessions == np.unique(sessions)[0]
    one_run = runs == np.unique(runs)[0]
    run_session = np.logical_and(one_session, one_run)
    len_run = sum(run_session * 1)

    model = init_model(n_chans, n_classes, input_window_samples, config=config)
    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, config)

    # Create pipeline
    create_dataset_with_align = TransformaParaWindowsDatasetEA(len_run)
    create_dataset = TransformaParaWindowsDataset()

    pipes = {}

    pipe_with_align = Pipeline([("Braindecode_dataset", create_dataset_with_align),
                                ("Net", clone(clf))])
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", clone(clf))])

    if args.ea == 'alignment':
        pipes[f"{config.model.type}_EA"] = pipe_with_align
        if config.model.type == "EEGNetv4":
            run = f'/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/output/runs/indiv_m1_final-{args.dataset}-alignment' \
                  '-exp_4-0-both'
        elif config.model.type == "ShallowFBCSPNet":
            run = f'/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/output/runs/indiv_m2_final-{args.dataset}-alignment' \
                  '-exp_4-0-both'
    else:
        pipes[f"{config.model.type}_Without_EA"] = pipe
        if config.model.type == "EEGNetv4":
            run = f'/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/output/runs/indiv_m1_final-{args.dataset}-no' \
                  '-alignment-exp_4-0-both'
        elif config.model.type == "ShallowFBCSPNet":
            run = (f'/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/output/runs/indiv_m2_final-{args.dataset}-no'
                   f'-alignment-exp_4-0-both')
    # Now, Online with 1 run for EA and ft
    results = online_indiv(dataset, paradigm, pipes, model, Path(run), config)

    print(results.head())

    # Save results
    results.to_csv(f"{run_dir}/{experiment_name}_fine-tuning_results_{config.ft.lr}.csv")


    print("---------------------------------------")

    # return results


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
