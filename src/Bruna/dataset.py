import copy
import pickle

from numpy import multiply
import numpy as np

from braindecode.datasets import MOABBDataset, create_from_X_y, BaseConcatDataset
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from braindecode.preprocessing import create_windows_from_events
import mne

from moabb.datasets import BNCI2014001, PhysionetMI
from moabb.paradigms import LeftRightImagery
from scipy.linalg import inv, sqrtm
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance

from sklearn.model_selection import StratifiedShuffleSplit

from model_validation import split_train_val, split_run, split_size
from alignment import euclidean_alignment
from riemann import riemannian_alignment, resting_alignment

mne.set_log_level("ERROR")


def read_dataset(config, dataset_name="BNCI2014001", subject_ids=None):
    """
    Read data from MOABB
    :param config: omegaconf object
    :param dataset_name: moabb data name
    :param subject_ids: range of subjects to load
    :return: windows data
    """
    try:
        file_task = open(f"{config.data}/{dataset_name}.pickle", "rb")

        windows_dataset = pickle.load(file_task)
    except:
        # TODO: add parameters to config file
        low_cut_hz = 0.5  # low cut frequency for filtering
        high_cut_hz = 38.0  # high cut frequency for filtering
        # Parameters for exponential moving standardization
        factor_new = 1e-3
        init_block_size = 1000
        factor = 1e6
        # Structure of the project

        if subject_ids is None:
            subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        dataset = MOABBDataset(
            dataset_name=dataset_name, subject_ids=subject_ids
        )

        preprocessors = [
            Preprocessor(
                "pick_types", eeg=True, meg=False, stim=False
            ),  # Keep EEG sensors
            Preprocessor(lambda data:
                         multiply(data, factor)),
            Preprocessor(
                "filter", l_freq=low_cut_hz, h_freq=high_cut_hz
            ),  # Bandpass filter
            Preprocessor(
                exponential_moving_standardize,  # Exponential moving standardization
                factor_new=factor_new,
                init_block_size=init_block_size,
            ),
        ]

        # Transform the data
        dataset = preprocess(dataset, preprocessors)

        trial_start_offset_seconds = -0.5
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )
        with open(f"{config.data}/{dataset_name}.pickle", "wb") as handle:
            pickle.dump(
                windows_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    return windows_dataset


def read_data(config, dataset_name="BNCI2014001", subject_ids=None):
    # Define Evaluation
    # For testing, try just with two classes
    paradigm = LeftRightImagery()

    # They are <moabb.datasets> and we want VaseConcatDataset
    if dataset_name == "BNCI2014001":
        # Because this is being auto-generated we only use 2 subjects
        dataset = BNCI2014001()

    else:
        dataset = PhysionetMI(imagined=True)

    return dataset


def create_ft_dataset(Train, Test_runs):
    # Now, we can concatenate the Train Data with the different sizes of the test data
    Train_before = Train

    Train_test = []
    for i in range(len(Test_runs)):
        test_runs = Test_runs[i]
        Train_concat = BaseConcatDataset([Train_before, test_runs])
        Train_test.append(Train_concat)

    return Train_test


def ft_test_data(test_subj, subject_ids, run, Data_subjects):
    # First we train for a specific subject as test
    # Define the test subject
    train_idx = copy.deepcopy(subject_ids)
    train_idx.remove(test_subj)

    # Test_data
    Test = Data_subjects[f'{test_subj}']
    Test_T, Test_E = split_train_val(Test, val_subj=None)
    # The session_T is going to be used in the training
    # The session_E is going to be used as test
    Data_run_T = split_run(Test_T, run)
    Test_runs = split_size(Data_run_T, run)
    return Test_runs, Test_T, Test_E


def split_runs_EA(X, len_run):
    X_aux = []
    m = len_run
    n = X.shape[0]
    for k in range(int(n / m)):
        run = X[k * m:(k + 1) * m]
        run_EA, _ = euclidean_alignment(run)
        X_aux.append(run_EA)
    X_EA = np.concatenate(X_aux)
    return X_EA


def split_runs_RA(X, len_run):
    X_aux = []
    m = len_run
    n = X.shape[0]
    for k in range(int(n / m)):
        run = X[k * m:(k + 1) * m]
        run_RA, _ = riemannian_alignment(run)
        X_aux.append(run_RA)
    X_RA = np.concatenate(X_aux)
    return X_RA


def split_runs_RS(X, tbreak, len_run):
    X_aux = []
    m = len_run
    n = X.shape[0]
    for k in range(int(n / m)):
        run = X[k * m:(k + 1) * m]
        run_RS, _ = resting_alignment(run, tbreak)
        X_aux.append(run_RS)
    X_RS = np.concatenate(X_aux)
    return X_RS


def split_runs_RS_v2(X, X_rest, domain):
    X_aux = []
    for d in np.unique(domain):
        X_d = X[domain == d]
        X_r = X_rest[d - 1]

        cov_r = covariances(X_r, estimator='cov')
        r = mean_covariance(cov_r, metric='riemann')
        r_op = inv(sqrtm(r))

        result_d = np.matmul(r_op, X_d)

        X_aux.append(result_d)
    X_RS = np.concatenate(X_aux)
    return X_RS


def delete_trials(X, y, subjects, seed, ea):
    subj = np.unique(subjects)
    train_idx = []
    l = []

    for i in range(len(subj)):
        s = subj[i]

        ix = subjects == s
        X_subj = X[ix]
        y_subj = y[ix]

        length = len(y_subj)
        l.append(length)

        n = ea

        p = (length - n * (length // n)) / length

        if p == 0:
            ix_train = np.where(ix)[0]
            train_idx.append(ix_train)

        if p != 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=p, random_state=seed)

            for j, (ix_train, ix_test) in enumerate(sss.split(X_subj, y_subj)):

                if i != 0:
                    ix_train = ix_train + i * l[i - 1]
                else:
                    ix_train = ix_train

            train_idx.append(ix_train)

    train_idx = np.concatenate(train_idx)

    return train_idx


def delete_trials(X, y, subjects, seed, ea):
    subj = np.unique(subjects)
    train_idx = []

    for s in subj:
        ix = subjects == s
        pos = np.where(ix)[0]

        X_subj = X[ix]
        y_subj = y[ix]

        length = len(y_subj)
        n = ea

        p = (length - n * (length // n)) / length

        if p == 0:
            train_idx.append(pos)

        # p!=0
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=p, random_state=seed)

            for j, (ix_train, ix_test) in enumerate(sss.split(X_subj, y_subj)):
                use = pos[ix_train]
            train_idx.append(use)

    train_idx = np.concatenate(train_idx)
    train_idx = np.sort(train_idx)

    return train_idx
