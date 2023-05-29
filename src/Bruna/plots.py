import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


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


# Average for each subject:
# Each column k corresponds the average of the scores obtained by testing subj k on all models
def boxplot_exp4_subj_avg(results1, results2):
    sns.set_theme()
    results = pd.concat([results1, results2])

    figbox, axbox = plt.subplots(figsize=(15, 5))

    plt.title(f"Average test subject accuracy", fontsize=15)

    axbox = sns.boxplot(data=results, y="score", x="test",
                        hue="pipeline", ax=axbox)
    axbox = sns.stripplot(data=results, y="score", x="test",
                          hue="pipeline", ax=axbox, dodge=True,
                          linewidth=1, alpha=.5)

    axbox.set_ylabel("Accuracy", fontsize=14.5)
    axbox.set_xlabel("Subjects", fontsize=14.5)

    figbox.savefig(f"box_test_avg.pdf", format='pdf', dpi=300, bbox_inches='tight')

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


def lineplot_exp2(results1, results2, subj_list):
    results = pd.concat([results1, results2])

    for subj in subj_list:
        results_subj = results[results['subject'] == subj]

        fig, ax = plt.subplots(figsize=(15, 5))

        plt.title(f"Score of test subject {subj} vs size of train set", fontsize=15)

        sns.lineplot(data=results_subj, x='n_train_runs', y='score', marker='+', hue='pipeline', ax=ax)
        ax.set_ylabel("Score", fontsize=14.5)
        ax.set_xlabel("Runs", fontsize=14.5)

        fig.savefig(f"exp2_subj_{subj}.pdf", format='pdf', dpi=300, bbox_inches='tight')

        plt.show()


def barplot_exp2_avg(results1, results2, subj_list):
    sns.set_theme()

    results = pd.concat([results1, results2])

    fig, ax = plt.subplots(figsize=(15, 5))
    plt.title("Average accuracy - different training sizes", fontsize=15)

    for subj in subj_list:
        results_subj = results[results['subject'] == subj]

        sns.lineplot(data=results_subj, x='n_train_runs', y='score', marker='+', hue='pipeline', ax=ax, alpha=0.15,
                     legend=False)

    sns.lineplot(data=results, x='n_train_runs', y='score', marker='+', hue='pipeline', ax=ax, alpha=1, legend=False)

    ax.set_ylabel("Score", fontsize=14.5)
    ax.set_xlabel("Runs", fontsize=14.5)

    fig.savefig(f"exp2_subj_with_avg.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()


def line_plot_exp3_avg(results1, results2, subj_list, type_):
    sns.set_theme()
    results = pd.concat([results1, results2])

    fig, ax = plt.subplots(figsize=(15, 5))

    for subj in subj_list:
        results_subj = results[results['subject'] == subj]

        sns.lineplot(data=results_subj, x='n_test_runs', alpha=0.15, legend=False, y='score', marker='+',
                     hue='pipeline', ax=ax)

    sns.lineplot(data=results, x='n_test_runs', y='score', alpha=1, legend=False, marker='+', hue='pipeline', ax=ax)

    ax.set_ylabel("Score", fontsize=14.5)
    ax.set_xlabel("Fine-tuning runs", fontsize=14.5)

    fig.savefig(f"{type_}_exp3_subj_avg.pdf", format='pdf', dpi=300, bbox_inches='tight')

    plt.show()
