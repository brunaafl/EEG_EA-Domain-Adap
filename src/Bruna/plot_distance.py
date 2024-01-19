import torch
from pathlib import Path

from moabb import set_download_dir
from moabb.datasets import Schirrmeister2017, BNCI2014001
from omegaconf import OmegaConf
from pyriemann.utils.covariance import covariances

from paradigm import MotorImagery_
from dataset import delete_trials
from plots import mean_group, distance_subjects
from util import parse_args, set_determinism, set_run_dir


def main(args):

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

    paradigm = MotorImagery_(events=events, channels=channels, n_classes=len(events), metric='accuracy', resample=250)

    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014001()
    elif args.dataset == 'Schirrmeister2017':
        dataset = Schirrmeister2017()

    X, y, metadata = paradigm.get_data(dataset=dataset, return_epochs=False)
    groups = metadata.subject.values
    # Delete some trials
    if dataset.code == "Schirrmeister2017":
        len_ea = 24
        train_idx = delete_trials(X, y, groups, 42, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]

    cov = covariances(X)

    means_subjects = mean_group(cov, domains=groups)
    path_plot = '/mnt/beegfs/home/aristimunha/bruna/EEG_EA-Domain-Adap/plot/distances/'

    distance_subjects(means_subjects, dataset, Path(path_plot))

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
