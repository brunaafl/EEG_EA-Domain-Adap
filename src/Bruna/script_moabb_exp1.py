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
        paradigm = LeftRightImagery(resample=100.0)

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

    create_dataset_with_align = TransformaParaWindowsDatasetEA(rpc, n_classes)
    create_dataset = TransformaParaWindowsDataset()

    pipes = {}

    pipe_with_align = Pipeline([("Braindecode_dataset", create_dataset_with_align),
                                ("Net", clone(clf))])
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", clone(clf))])

    if args.ea == 'alignment':
        pipes["EEGNetv4_EA"] = pipe_with_align
    else:
        pipes["EEGNetv4_Without_EA"] = pipe

    # Define evaluation and train
    overwrite = False  # set to True if we want to overwrite cached results
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        suffix=f"experiment_1_{args.dataset}",
        overwrite=overwrite,
        return_epochs=True,
        hdf5_path=run_dir,
        n_jobs=-1,
    )

    results = evaluation.process(pipes)
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
