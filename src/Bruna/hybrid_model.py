import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skorch.dataset import ValidSplit, unpack_data
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.callbacks.scoring import _cache_net_forward_iter
from skorch.utils import to_tensor

import numpy as np
import pandas as pd

from braindecode.models import EEGNetv4
from braindecode import EEGClassifier
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows

from sklearn.model_selection import (
	LeaveOneGroupOut,
	StratifiedKFold,
	StratifiedShuffleSplit,
	cross_val_score,
)
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from moabb.evaluations.base import BaseEvaluation

from time import time
from typing import Union

from copy import deepcopy

from mne.epochs import BaseEpochs
import mne

import pdb

from torchviz import make_dot

import random

import traceback

from torch.utils.data.dataloader import default_collate

def gen_slice_EEGNet(n_chans, n_classes, input_window_samples, config, start=0, end=19, drop_prob=0.5, remove_bn=True):

	temp_model = EEGNetv4(
		n_chans,
		n_classes,
		input_window_samples=input_window_samples,
		final_conv_length=config.model.final_conv_length,
		drop_prob=config.model.drop_prob
	)

	if remove_bn:
		for i, module in enumerate(temp_model):
			if isinstance(temp_model[i], nn.BatchNorm2d):
				temp_model[i] = nn.Identity()

	return nn.Sequential(*(list(temp_model.children())[start:end]))

class HybridModel(nn.Module):
	def __init__(self, num_models, n_chans, n_classes, input_window_samples, config=None):
		super(HybridModel, self).__init__()
		self.num_models = num_models
		self.shared_modules = gen_slice_EEGNet(n_chans, n_classes, input_window_samples, config, start=6)
		self.unique_modules = nn.ModuleList()
		for model in range(num_models):
			self.unique_modules.append(self.init_unique_modules(n_chans, n_classes, input_window_samples, config))

	def init_unique_modules(self, n_chans, n_classes, input_window_samples, config):
		unique_head = gen_slice_EEGNet(n_chans, n_classes, input_window_samples, config, end=5, remove_bn=False)
		# , nn.LayerNorm((16, 1, 1126), elementwise_affine=False)
		return unique_head

	def split_input(self, X):
		return torch.split(X, int(X.shape[1]/self.num_models), dim=1)

	def forward(self, x):
		inputs = self.split_input(x)
		out = []
		for i, model_input in enumerate(inputs):
			temp_unique = self.unique_modules[i](model_input)
			temp_shared = self.shared_modules(temp_unique)
			#pdb.set_trace()
			out.append(temp_shared)
		result = torch.stack(out)
		if result.requires_grad:
			result.retain_grad()
		return result

	def predict(self, X):
		return [out.argmax(axis=1) for out in self.forward(X)]


class HybridClassifier(EEGClassifier):
	def get_loss(self, y_pred, y_true, *args, **kwargs):
		#if isinstance(self.criterion_, nn.NLLLoss):
		#	eps = torch.finfo(y_pred.dtype).eps
		#	y_pred = torch.log(y_pred + eps)
		y_true = to_tensor(y_true, device=self.device)
		losses = []
		for subject in range(y_pred.shape[0]):
			subject_slice = torch.select(y_pred, 0, subject)
			if y_pred.requires_grad:
				subject_slice.retain_grad()
			losses.append(self.criterion_(subject_slice, y_true[:, subject]))

		loss = sum(losses) / self.module.num_models

		#pdb.set_trace()
		#loss.backward()
		make_dot(y_pred, show_attrs=True, params=dict(self.module.named_parameters())).render("model", format="svg")
		#pdb.set_trace()

		return loss


class HybridScoring(EpochScoring):
	def on_epoch_begin(self, net, dataset_train, dataset_valid, **kwargs):
		self.y_preds_ = []
		self.y_trues_ = []
		for subject_i in range(net.module.num_models):
			self.y_preds_.append([])

	def on_batch_end(
			self, net, batch, y_pred, training, **kwargs):
		if not self.use_caching or training != self.on_train:
			return

		_X, y = unpack_data(batch)
		self.y_trues_.append(y)
		for subject_i in range(net.module.num_models):
			self.y_preds_[subject_i].append(torch.select(y_pred, 0, subject_i))

	def on_epoch_end(
			self,
			net,
			dataset_train,
			dataset_valid,
			**kwargs):
		X_test, y_test, y_pred = self.get_test_data(dataset_train, dataset_valid)
		
		unwrapped_y_pred = []
		for subject_i in range(net.module.num_models):
			unwrapped_y_pred.append(torch.vstack(y_pred[subject_i]))

		with _cache_net_forward_iter(net, self.use_caching, unwrapped_y_pred) as cached_net:
			current_score = self._scoring(cached_net, X_test, y_test)

		self._record_score(net.history, current_score)

def get_subject_acc_scorer(subject):
	def scoring_for_subject_i(model, x, y_true):
		out = model.forward_iter()
		y_preds = [z for z in out]
		subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
		true_slice = y_true[:, subject]
		predictions = np.argmax(subject_slice, axis=1)
		return accuracy_score(true_slice, predictions)
	return scoring_for_subject_i

def average_acc_scoring(model, x, y_true):
	out = model.forward_iter()
	y_preds = [z for z in out]
	accuracies_per_subject = []
	for subject in range(len(y_preds)):
		subject_slice = np.exp(y_preds[subject].detach().cpu().numpy())
		true_slice = y_true[:, subject]
		predictions = np.argmax(subject_slice, axis=1)
		accuracies_per_subject.append(accuracy_score(true_slice, predictions))
	return sum(accuracies_per_subject)/len(accuracies_per_subject)


def define_hybrid_clf(model, config):
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

	lrscheduler = LRScheduler(policy='StepLR', step_size=30, gamma=0.1)

	scoring_callbacks = [HybridScoring(scoring=get_subject_acc_scorer(i), on_train=False, name=f'{i}_valid_acc', lower_is_better=False) for i in range(model.num_models)]

	clf = HybridClassifier(
		model,
		criterion=torch.nn.NLLLoss,
		optimizer=torch.optim.AdamW,
		train_split=ValidSplit(config.train.valid_split, random_state=config.seed),
		optimizer__lr=lr,
		optimizer__weight_decay=weight_decay,
		batch_size=batch_size,
		max_epochs=config.train.n_epochs,
		callbacks=[EarlyStopping(monitor='valid_loss', patience=patience),
		HybridScoring(scoring=average_acc_scoring, on_train=True, name='avg_train_acc', lower_is_better=False),
		HybridScoring(scoring=average_acc_scoring, on_train=False, name='avg_valid_acc', lower_is_better=False)] + scoring_callbacks, #, lrscheduler],
		device=device,
		verbose=1,
	)
	return clf


class HybridAggregateTransform(BaseEstimator, TransformerMixin):
	def __init__(self, kw_args=None):
		self.kw_args = kw_args

	def fit(self, X, y=None, subject_groups=None, info=None, labels=None):
		self.labels = labels
		self.groups = subject_groups
		self.info = info
		return self

	def transform(self, X, y=None):
		subjects = {i:[] for i in np.unique(self.groups)}

		for index, trial in enumerate(X):
			subjects[self.groups[index]].append((trial, self.labels[index]))

		assert len(np.unique([len(subjects[i]) for i in subjects])) == 1

		n_subjects = len(subjects)
		n_trials_per_subject = len(subjects[list(subjects.keys())[0]])
		ch_names = [self.info["ch_names"][i] + f"_s{k}" for i in range(len(self.info["ch_names"])) for k in subjects.keys()]

		new_trials = []
		for trial_i in range(n_trials_per_subject):
			trial = []
			target = []
			for subject in subjects:
				trial.append(subjects[subject][trial_i][0])
				target.append(subjects[subject][trial_i][1])
			info = mne.create_info(ch_names=ch_names, sfreq=self.info["sfreq"])
			raw = mne.io.RawArray(np.vstack(trial), info)
			base_dataset = BaseDataset(raw, pd.Series({"target": np.array(target)}), target_name="target")
			new_trials.append(base_dataset)

		dataset = BaseConcatDataset(new_trials)
		windows_dataset = create_fixed_length_windows(
			dataset,
			start_offset_samples=0,
			stop_offset_samples=None,
			window_size_samples=len(new_trials[0]),
			window_stride_samples=len(new_trials[0]),
			drop_last_window=False
		)


		return windows_dataset

	def __sklearn_is_fitted__(self):
		"""Return True since Transfomer is stateless."""
		return True


class HybridEvaluation(BaseEvaluation):
	def is_valid(self, dataset):
		return len(dataset.subject_list) > 1	
	def evaluate(self, dataset, pipelines, grid_search):
		"""Evaluate results on a single dataset.

		This method return a generator. each results item is a dict with
		the following convension::

			res = {'time': Duration of the training ,
				   'dataset': dataset id,
				   'subject': subject id,
				   'session': session id,
				   'score': score,
				   'n_samples': number of training examples,
				   'n_channels': number of channel,
				   'pipeline': pipeline name}
		"""
		
		X, y, metadata = self.paradigm.get_data(dataset, return_epochs=self.return_epochs)

		# encode labels
		le = LabelEncoder()
		y = y if self.mne_labels else le.fit_transform(y)

		# extract metadata
		groups = metadata.subject.values
		sessions = metadata.session.values
		n_subjects = len(dataset.subject_list)

		scorer = get_scorer(self.paradigm.scoring)

		cv = LeaveOneGroupOut()
		# Progressbar at subject level
		for train, test in tqdm(cv.split(X, y, groups), total=n_subjects, desc=f"{dataset.code}-CrossSubject",):
			subject = groups[test[0]]
			# now we can check if this subject has results
			run_pipes = self.results.not_yet_computed(pipelines, dataset, subject)

			# iterate over pipelines
			for name, clf in run_pipes.items():
				t_start = time()
				model = deepcopy(clf).fit(X[train], None, Hybrid_adapter__labels=y[train], Hybrid_adapter__subject_groups=groups[train], Hybrid_adapter__info=X[train].info)
				duration = time() - t_start

				# we eval on each session
				for session in np.unique(sessions[test]):
					ix = sessions[test] == session
					score = _score(model, X[test[ix]], y[test[ix]], scorer)

					nchan = (
						X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
					)
					res = {
						"time": duration,
						"dataset": dataset,
						"subject": subject,
						"session": session,
						"score": score,
						"n_samples": len(train),
						"n_channels": nchan,
						"pipeline": name,
					}

					yield res