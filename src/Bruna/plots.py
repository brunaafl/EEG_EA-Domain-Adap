from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd
from pyriemann.utils.covariance import covariances
from pyriemann.utils.distance import pairwise_distance

from moabb.datasets import BNCI2014001, Schirrmeister2017
from pyriemann.utils.mean import mean_covariance

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from paradigm import MotorImagery_
from dataset import delete_trials


def statistical_analysis(results1_csv, results2_csv, exp_name):
    """
    Execute statistical analysis and plot graphs using MOABB functions

    :param results2_csv: csv file
    :param results1_csv: csv file
    :param args:
    :return:
    """

    # With EA
    results1 = pd.read_csv(results1_csv)
    # Without EA
    results2 = pd.read_csv(results2_csv)

    # Set run dir
    # config = OmegaConf.load(args.config_file)
    # run_dir, experiment_name = set_run_dir(config, args)

    # Concat results from different pipelines
    results = pd.concat([results1, results2])

    fig, color_dict = moabb_plt.score_plot(results)
    fig.savefig(f"{exp_name}_score_plot_models.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot paired scores
    fig = moabb_plt.paired_plot(results, "EEGNetv4_EA", "EEGNetv4_Without_EA")
    fig.savefig(f"{exp_name}_paired_score_plot_models.pdf", format='pdf', dpi=300, bbox_inches='tight')

    # Compute p values using permutation test
    stats = compute_dataset_statistics(results)
    P, T = find_significant_differences(stats)

    fig = moabb_plt.meta_analysis_plot(stats, "EEGNetv4_EA", "EEGNetv4_Without_EA")
    fig.savefig(f"{exp_name}_meta_analysis_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # plot meta analysis summary
    fig = moabb_plt.summary_plot(P, T)
    fig.savefig(f"{exp_name}_meta_analysis_summary_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def boxplot_exp1(results1_csv, results2_csv):
    # With EA
    results1 = pd.read_csv(results1_csv)
    # Without EA
    results2 = pd.read_csv(results2_csv)

    results = pd.concat([results1, results2])

    sns.set_theme()

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title(f"Score of each test individual in the shared model", fontsize=15)

    ax = sns.boxplot(data=results, y="score", x="subject",
                     hue="pipeline", ax=ax)

    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_ylabel("Score", fontsize=14.5)
    ax.set_xlabel("Subjects", fontsize=14.5)

    fig.savefig("boxplot_shared.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    figbar, axbar = plt.subplots(figsize=(15, 5))
    plt.title(f"Score of each test individual in the shared model", fontsize=15)

    axbar = sns.barplot(data=results, y="score", x="subject",
                        hue="pipeline", ax=axbar)

    # sns.move_legend(axbar, "upper left", bbox_to_anchor=(1, 1))
    axbar.set_ylabel("Score", fontsize=14.5)
    axbar.set_xlabel("Subjects", fontsize=14.5)

    figbar.savefig("bar_shared.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


# Barplot per test subject:
# Each image corresponds to the score values for a fixed subject tested on all models
def test_barplot_exp4(results1, results2, subj_list):
    results = pd.concat([results1, results2])

    for subj in subj_list:
        results_subj = results[results['subject'] == subj]

        figbar, axbar = plt.subplots(figsize=(15, 5))
        plt.title(f"Score of test subject {subj} in each individual model", fontsize=15)

        axbar = sns.barplot(data=results_subj, y="score", x="test",
                            hue="pipeline", ax=axbar)

        # sns.move_legend(axbar, "upper left", bbox_to_anchor=(1, 1))
        axbar.set_ylabel("Score", fontsize=14.5)
        axbar.set_xlabel("Model", fontsize=14.5)

        figbar.savefig(f"bar_individual_{subj}.pdf", format='pdf', dpi=300, bbox_inches='tight')

        plt.show()


# Average for each subject:
# Each column k corresponds the average of the scores obtained by testing subj k on all models
def avg_barplot_exp4(results1, results2, subj_list):
    data = []
    for subj in subj_list:
        mean_EA = results1[results1['test'] == subj]['score'].mean()
        mean_no_EA = results2[results2['test'] == subj]['score'].mean()

        data.append([subj, mean_EA, 'EEGNetv4_EA'])
        data.append([subj, mean_no_EA, 'EEGNetv4_Without_EA'])

    df = pd.DataFrame(data, columns=['test_subj', 'avg_score', 'pipeline'])

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title(f"Average accuracy of each subject in different individual models", fontsize=15)

    ax = sns.barplot(data=df, y="avg_score", x="test_subj",
                     hue="pipeline", ax=ax)

    # sns.move_legend(axbar, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylabel("Accuracy", fontsize=14.5)
    ax.set_xlabel("Subjects", fontsize=14.5)

    fig.savefig(f"bar_avg_individual.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


# Average for each model:
# Each column k corresponds the average of the scores obtained by testing all subj on model k
def boxplot_exp4_model_avg(results1, results2):
    sns.set_theme()
    results = pd.concat([results1, results2])

    figbox, axbox = plt.subplots(figsize=(15, 5))

    plt.title(f"Average model accuracy", fontsize=15)

    axbox = sns.boxplot(data=results, y="score", x="subject",
                        hue="pipeline", ax=axbox)
    axbox = sns.stripplot(data=results, y="score", x="subject",
                          hue="pipeline", ax=axbox, dodge=True,
                          linewidth=1, alpha=.5)

    axbox.set_ylabel("Accuracy", fontsize=14.5)
    axbox.set_xlabel("Models", fontsize=14.5)

    figbox.savefig(f"box_model_avg.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


# Heatplot for exp4
def heatplot(resultsEA, results, n_subj):
    # Prepare data
    sujeitos_id = list(range(1, n_subj + 1))

    types = ['EA', 'No_EA']

    for type_ in types:

        if type_ == "EA":
            res = resultsEA
        else:
            res = results

        table = []

        for k in sujeitos_id:
            model_k = []

            table_model_k = res[res['subject'] == k]['score'].reset_index()

            for i in sujeitos_id:
                if i == k:
                    model_k.append(0)
                else:
                    if i < k:
                        model_k.append(table_model_k['score'][i - 1] * 100)
                    else:
                        model_k.append(table_model_k['score'][i - 2] * 100)

            N = len(model_k)
            mean = sum(model_k) / (N - 1)
            model_k.append(mean)
            table.append(model_k)

        n = len(table[:][0])
        mean = []

        for i in range(n - 1):
            aux = table[:][i]

            mean_sum = sum(aux) / n

            mean.append(mean_sum)

        r = 0
        for i in range(n - 1):
            r = r + table[i][9]

        mean.append(r)
        table.append(mean)
        table = np.array(table, dtype=object)

        fig, axes = plt.subplots()

        hm = sns.heatmap(data=table, annot=True, annot_kws={"fontsize": 9}, fmt=".1f", cmap="crest")  #
        hm.set_xticklabels(sujeitos_id)
        hm.set_yticklabels(sujeitos_id)

        if type_ == "EA":
            title = "Transferability - EEGNet with EA"
            name = "transferability_EA.pdf"
        else:
            title = "Transferability - EEGNet without EA"
            name = "transferability_no_EA.pdf"

        plt.title(title, fontsize=14)

        axes.set_ylabel('Model')

        axes.set_xlabel('Test Subject')

        plt.show()

        fig.savefig(name, format='pdf', dpi=300, bbox_inches='tight')


def distance_subjects(list_mean, dataset, path):
    distance = pairwise_distance(np.array(list_mean), np.array(list_mean))
    distance[distance < 1e-10] = 0

    fig, ax = plt.subplots()
    # distance = 1/max*distance

    df = pd.DataFrame(distance, columns=dataset.subject_list, index=dataset.subject_list)
    # Plot heatmap
    sns.heatmap(data=df, cmap=sns.cubehelix_palette(as_cmap=True),
                annot=True, )  # sns.cubehelix_palette(as_cmap=True)

    fig.savefig(path / f"Mean_distance_{dataset.code}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def mean_group(covmats, size=24, domains=None, metric='riemann'):
    # Returns a list containing the mean of each batch/group of matrices
    covmats_means = []
    if domains is None:
        m = size
        n = covmats.shape[0]
        for k in range(int(n / m)):
            cov_batch = covmats[k * m:(k + 1) * m]
            cov_batch_mean = mean_covariance(cov_batch, metric=metric)
            covmats_means.append(cov_batch_mean)
    else:
        for d in np.unique(domains):
            cov_batch = covmats[domains == d]
            cov_batch_mean = mean_covariance(cov_batch, metric=metric)
            covmats_means.append(cov_batch_mean)
    return covmats_means


datsets = [BNCI2014001(), Schirrmeister2017()]
for data in datsets:

    events = ["right_hand", "left_hand"]
    channels = ["Fz", "FC3", "FCz", "FC4", "C5", "FC1", "FC2",
                "C3", "C4", "Cz", "C6", "CPz", "C1", "C2",
                "CP2", "CP1", "CP4", "CP3", "Pz", "P2", "P1", "POz"]

    paradigm = MotorImagery_(events=events, channels=channels, n_classes=len(events), metric='accuracy', resample=250)
    X, y, metadata = paradigm.get_data(dataset=data, return_epochs=False)
    groups = metadata.subject.values
    # Delete some trials
    if data.code == "Schirrmeister2017":
        len_ea = 24
        train_idx = delete_trials(X, y, groups, 42, len_ea)
        X = X[train_idx]
        y = y[train_idx]
        groups = groups[train_idx]

    cov = covariances(X)

    means_subjects = mean_group(cov, domains=groups)
    root = Path(__file__).parent.parent
    path_plot = root / 'plot' / 'distances'

    distance_subjects(means_subjects, data, path_plot)
