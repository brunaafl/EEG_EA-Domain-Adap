"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""

import torch
from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import MotorImagery, LeftRightImagery

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
import matplotlib.pyplot as plt


from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from moabb.utils import set_download_dir

from pipeline import ClassifierModel, TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
from train import define_clf, init_model
from util import parse_args, set_determinism, set_run_dir

from hybrid_model import HybridModel, HybridEvaluation, HybridAggregateTransform, define_hybrid_clf, gen_slice_DeepNet


import torchinfo

import numpy as np

import pdb

from time import time

from braindecode.models import EEGNetv4, Deep4Net

"""
For the joint model
"""


def main(args):
    """
    Parameters
    ----------
    args : object
    """
    torch.set_num_threads(1)

    init_time = time()

    config = OmegaConf.load(args.config_file)
    eval_config = OmegaConf.load(args.eval_config_file)
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
    if args.dataset == 'Schirrmeister2017':
        ch = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP6", "CP4", "CP2"]
    else:
        ch = None
    paradigm = MotorImagery(events=events, n_classes=len(events), channels=ch)

    print(f"(1) Initial {(time() - init_time) * 1000}ms | {(time() - init_time)}s")


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
        paradigm = LeftRightImagery(resample=100.0)

    datasets = [dataset]
    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    n_chans = X.shape[1]
    input_window_samples = X.shape[2]
    rpc = len(meta['session'].unique())*len(meta['run'].unique())

    print(f"(2) Get Data Done {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    num_subjects = len(dataset.subject_list)

    #testmodel = gen_slice_DeepNet(n_chans, n_classes, input_window_samples, config, drop_prob=config.model.drop_prob)
    #pdb.set_trace()

    model = HybridModel(num_subjects - 1, n_chans, n_classes, input_window_samples, config=config, freeze=args.freeze)
    # Send model to GPU
    if cuda:
        model.cuda()


    # Create Classifier
    clf = define_hybrid_clf(model, config)

    print(f"(3) Created clf {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    create_dataset_with_align = TransformaParaWindowsDatasetEA(rpc, n_classes)
    create_dataset = TransformaParaWindowsDataset()

    runs = meta.run.values
    sessions = meta.session.values
    one_session = sessions == np.unique(sessions)[0]
    one_run = runs == np.unique(runs)[0]
    run_session = np.logical_and(one_session, one_run)
    len_run = sum(run_session * 1)

    hybrid_adapter = HybridAggregateTransform()
    hybrid_adapter_EA = HybridAggregateTransform(EA_len_run=len_run)

    pipes = {}

    pipe_with_align = Pipeline([("Hybrid_adapter", hybrid_adapter_EA),
                                ("Net", clone(clf))])
    pipe = Pipeline([("Hybrid_adapter", hybrid_adapter),
                     ("Net", clone(clf))])

    freeze_tag = ["-Frozen", "-Not_Frozen"][args.freeze == "no-freeze"]

    if args.ea == 'alignment':
        pipes["EEGNetv4_EA" + freeze_tag] = pipe_with_align
    else:
        pipes["EEGNetv4_Without_EA" + freeze_tag] = pipe

    print(f"(4) Before eval setup {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    # Define evaluation and train
    overwrite = False  # set to True if we want to overwrite cached results
    evaluation = HybridEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        suffix=f"experiment_1_{args.dataset}",
        overwrite=overwrite,
        return_epochs=True,
        hdf5_path=run_dir,
        n_jobs=-1,
        eval_config=eval_config,
        EA_in_eval=(args.ea == 'alignment'),
        len_run=len_run,
    )

    print(f"(5) Before eval {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    results = evaluation.process(pipes)
    print(results.head())

    # Save results
    results.to_csv(f"{run_dir}/{experiment_name}_results.csv")


    print("---------------------------------------")

    # return results


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
