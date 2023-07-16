import copy

import torch

from braindecode import EEGClassifier
from braindecode.models import EEGNetv4, Deep4Net, ShallowFBCSPNet
from sklearn.metrics import roc_auc_score
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import ValidSplit
from skorch.helper import SliceDataset


def define_clf(model, config):
    """
    Transform the pytorch model into classifier object to be used in the training
    Parameters
    ----------
    model: pytorch model
    device: cuda or cpu
    config: dict with the configuration parameters
    Returns
    -------
    clf: skorch classifier
    """

    weight_decay = config.train.weight_decay
    batch_size = config.train.batch_size
    lr = config.train.lr
    patience = config.train.patience
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lrscheduler = LRScheduler(policy='CosineAnnealingLR', T_max=config.train.n_epochs, eta_min=0)

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=config.train.n_epochs,
        callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
                   lrscheduler,
                   EpochScoring(scoring='accuracy', on_train=True,
                                name='train_acc', lower_is_better=False),
                   EpochScoring(scoring='accuracy', on_train=False,
                                name='valid_acc', lower_is_better=False)],
        device=device,
        verbose=1,
    )

    clf.initialize()
    return clf


def clf_tuning(model, config):

    batch_size = config.train.batch_size
    n_epochs = config.train.n_epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Classifier
    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        optimizer__lr=[],
        optimizer__weight_decay=[],
        batch_size=batch_size,
        max_epochs=n_epochs,
        train_split=ValidSplit(config.train.valid_split, random_state=config.seed),  # train /test split is handled by GridSearchCV
        callbacks=[
            "accuracy",
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )

    return clf

def init_model(n_chans, n_classes, input_window_samples, config):
    if config.model.type == "Deep4Net":
        model = Deep4Net(
            in_chans=n_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=5,
            n_filters_spat=5,
            stride_before_pool=True,
            n_filters_2=int(2 ** 2),
            n_filters_3=int(2 ** 3),
            n_filters_4=int(2 ** 3),
            final_conv_length=config.model.final_conv_length,
            drop_prob=config.model.drop_prob
        )

    elif config.model.type == 'ShallowFBCSPNet':
        model = ShallowFBCSPNet(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=config.model.final_conv_length,
            drop_prob=config.model.drop_prob
        )

    else:
        model = EEGNetv4(
            n_chans,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=config.model.final_conv_length,
            drop_prob=config.model.drop_prob
        )
    return model


def fine_tuning(model, device, subj, Test, Train_test):
    bac_runs = []
    for i in range(len(Train_test)):
        # model, device = init_model()

        # Then, we initialize a new model
        # val_set = Val
        n_epochs = 100
        weight_decay = 0
        lr = 0.0625 * 0.01

        clf_fine_tune = EEGClassifier(
            module=copy.deepcopy(model),
            train_split=None,  # predefined_split(val_set)
            callbacks=[
                "accuracy", ("lr_scheduler",
                             LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ],
            device=device,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay
        )

        clf_fine_tune.initialize()  # This is important!

        clf_fine_tune.load_params(
            f_params=f'final_model_params_{subj}.pkl',
            f_optimizer=f'final_model_optimizer_{subj}.pkl',
            f_history=f'final_model_history_{subj}.json',
            f_criterion=f"final_model_criterion_{subj}.pkl")

        # Fit the new model
        clf_fine_tune.fit(Train_test[i], y=None, epochs=60)

        y_pred2 = clf_fine_tune.predict(Test)
        y_true2 = list(SliceDataset(Test, 1))
        bac2 = roc_auc_score(y_true=y_true2, y_pred=y_pred2)
        bac_runs.append(bac2)
        print(f"With {i + 1} test runs in the train : ")
        print(f"  bac = {bac2}")

        return bac_runs
