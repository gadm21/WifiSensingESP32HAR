"""
Federated Learning components for CSI-based activity recognition.

Contains:
- FedXgbBagging: Federated XGBoost bagging aggregation strategy
- Helper functions for FL simulation with FederatedPartitioner
- Verbose logging utilities
"""

import json
import time
from collections.abc import Callable
from logging import WARNING
from typing import Any, cast

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from utils import (compute_all_metrics, print_metrics_summary, set_global_seed,
                  METRICS_CSV_FIELDS, aggregate_seed_metrics)


# =============================================================================
# FedXgbBagging Strategy (from Flower)
# =============================================================================
def aggregate_xgb_trees(bst_prev_org: bytes | None, bst_curr_org: bytes) -> bytes:
    """Conduct bagging aggregation for given XGBoost trees.
    
    Combines trees from multiple clients by appending them to the global model.
    
    Parameters
    ----------
    bst_prev_org : bytes or None
        Previous global model as bytes. None for first round.
    bst_curr_org : bytes
        Current client model as bytes.
    
    Returns
    -------
    bytes
        Aggregated model.
    """
    if not bst_prev_org:
        return bst_curr_org

    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)

    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")
    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> tuple[int, int]:
    """Get number of trees and parallel trees from XGBoost model."""
    xgb_model = json.loads(bytearray(xgb_model_org))
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


# =============================================================================
# FL Simulation for XGBoost with FederatedPartitioner
# =============================================================================
class FedXGBoostSimulator:
    """Simulates Federated XGBoost training using FederatedPartitioner.
    
    Implements FedXgbBagging strategy in a simulation environment.
    Each partition represents a client with local data.
    
    Parameters
    ----------
    partitioner : FederatedPartitioner
        Partitioner containing the federated data splits.
    num_rounds : int
        Number of federated rounds. Default: 5
    local_epochs : int
        Number of local boosting rounds per client per FL round. Default: 1
    xgb_params : dict
        XGBoost parameters. Default: sensible defaults for classification.
    test_dataset : TrainingDataset, optional
        Held-out test set for global evaluation.
    verbose : bool
        Enable verbose logging. Default: True
    """
    
    def __init__(
        self,
        partitioner,
        num_rounds=5,
        local_epochs=1,
        xgb_params=None,
        test_dataset=None,
        verbose=True,
    ):
        self.partitioner = partitioner
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.test_dataset = test_dataset
        self.verbose = verbose
        
        self.xgb_params = xgb_params or {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'max_depth': 4,
            'eta': 0.1,
            'num_parallel_tree': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        
        self.global_model: bytes | None = None
        self.history = {
            'round': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'test_f1': [],
            'num_trees': [],
            'round_time': [],
        }
    
    def _log(self, msg, level='INFO'):
        """Print verbose log message."""
        if self.verbose:
            print(f"[FL-XGB] [{level}] {msg}")
    
    def _log_separator(self, char='=', length=70):
        if self.verbose:
            print(char * length)
    
    def _client_train(self, partition_id: int, global_round: int) -> tuple[bytes, int]:
        """Train a single client and return updated model.
        
        Parameters
        ----------
        partition_id : int
            Client/partition ID.
        global_round : int
            Current FL round (1-indexed).
        
        Returns
        -------
        tuple[bytes, int]
            (local_model_bytes, num_samples)
        """
        partition = self.partitioner.load_partition(partition_id)
        X, y = partition.X, partition.y
        num_samples = len(X)
        
        # Ensure num_class is set for multi-class (use global class count, not local)
        params = self.xgb_params.copy()
        num_classes = len(self.partitioner.dataset.label_map)
        if num_classes > 2:
            params['num_class'] = num_classes
        
        dtrain = xgb.DMatrix(X, label=y)
        
        if global_round == 1:
            # First round: train from scratch
            bst = xgb.train(
                params,
                dtrain,
                num_boost_round=self.local_epochs,
                verbose_eval=False,
            )
        else:
            # Subsequent rounds: load global model and continue training
            bst = xgb.Booster(params=params)
            bst.load_model(bytearray(self.global_model))
            
            # Local training: update with new trees
            for _ in range(self.local_epochs):
                bst.update(dtrain, bst.num_boosted_rounds())
            
            # Extract only the new trees for aggregation
            bst = bst[
                bst.num_boosted_rounds() - self.local_epochs : bst.num_boosted_rounds()
            ]
        
        local_model = bst.save_raw("json")
        return local_model, num_samples
    
    def _aggregate_round(self, client_models: list[tuple[bytes, int]]) -> None:
        """Aggregate client models using bagging.
        
        Parameters
        ----------
        client_models : list of (model_bytes, num_samples)
        """
        for model_bytes, _ in client_models:
            self.global_model = aggregate_xgb_trees(self.global_model, model_bytes)
    
    def _evaluate_global(self, X: np.ndarray, y: np.ndarray, full_metrics=False) -> dict:
        """Evaluate global model on given data.

        Parameters
        ----------
        full_metrics : bool
            If True, use compute_all_metrics for the full unified metric set.
        """
        if self.global_model is None:
            return {'accuracy': 0.0, 'f1': 0.0}
        
        # Load global model
        params = self.xgb_params.copy()
        num_classes = len(np.unique(y))
        if num_classes > 2:
            params['num_class'] = num_classes
            
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(self.global_model))
        
        dtest = xgb.DMatrix(X)
        y_pred_raw = bst.predict(dtest)
        
        # Handle multi-class vs binary
        y_prob = None
        if len(y_pred_raw.shape) > 1:
            y_prob = y_pred_raw  # already (N, C) softmax probs
            y_pred = np.argmax(y_pred_raw, axis=1)
        else:
            y_pred = y_pred_raw.astype(int)
        
        if full_metrics:
            metrics = compute_all_metrics(y, y_pred, y_prob=y_prob, n_classes=num_classes)
            metrics['predictions'] = y_pred
            return metrics

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        return {'accuracy': acc, 'f1': f1, 'predictions': y_pred}
    
    def run(self) -> dict:
        """Run the federated learning simulation.
        
        Returns
        -------
        dict
            Training history and final metrics.
        """
        num_clients = self.partitioner.num_partitions
        
        self._log_separator()
        self._log(f"Starting Federated XGBoost Simulation")
        self._log_separator()
        self._log(f"Clients: {num_clients}")
        self._log(f"FL Rounds: {self.num_rounds}")
        self._log(f"Local Epochs: {self.local_epochs}")
        self._log(f"XGB Params: {self.xgb_params}")
        self._log_separator('-')
        
        total_start = time.time()
        
        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()
            self._log(f"\n{'='*20} ROUND {round_num}/{self.num_rounds} {'='*20}")
            
            # Client training
            client_models = []
            total_samples = 0
            
            for client_id in range(num_clients):
                client_start = time.time()
                model_bytes, num_samples = self._client_train(client_id, round_num)
                client_time = time.time() - client_start
                
                client_models.append((model_bytes, num_samples))
                total_samples += num_samples
                
                self._log(
                    f"  Client {client_id}: {num_samples} samples, "
                    f"trained in {client_time:.2f}s",
                    level='DEBUG' if not self.verbose else 'INFO'
                )
            
            # Aggregation
            agg_start = time.time()
            self._aggregate_round(client_models)
            agg_time = time.time() - agg_start
            
            # Get tree count
            if self.global_model:
                num_trees, _ = _get_tree_nums(self.global_model)
            else:
                num_trees = 0
            
            self._log(f"  Aggregated {num_clients} clients in {agg_time:.2f}s")
            self._log(f"  Global model: {num_trees} trees")
            
            # Evaluation
            train_acc = 0.0
            if round_num == self.num_rounds or round_num % max(1, self.num_rounds // 3) == 0:
                # Evaluate on all training data
                all_X = []
                all_y = []
                for cid in range(num_clients):
                    part = self.partitioner.load_partition(cid)
                    all_X.append(part.X)
                    all_y.append(part.y)
                X_train_all = np.concatenate(all_X)
                y_train_all = np.concatenate(all_y)
                
                train_metrics = self._evaluate_global(X_train_all, y_train_all)
                train_acc = train_metrics['accuracy']
                self._log(f"  Train Accuracy: {train_acc:.4f}")
            
            test_acc = 0.0
            test_f1 = 0.0
            if self.test_dataset is not None:
                test_metrics = self._evaluate_global(self.test_dataset.X, self.test_dataset.y)
                test_acc = test_metrics['accuracy']
                test_f1 = test_metrics['f1']
                self._log(f"  Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            round_time = time.time() - round_start
            self._log(f"  Round time: {round_time:.2f}s")
            
            # Record history
            self.history['round'].append(round_num)
            self.history['train_accuracy'].append(train_acc)
            self.history['test_accuracy'].append(test_acc)
            self.history['test_f1'].append(test_f1)
            self.history['num_trees'].append(num_trees)
            self.history['round_time'].append(round_time)
        
        total_time = time.time() - total_start
        
        # Final evaluation
        self._log_separator()
        self._log("FINAL RESULTS")
        self._log_separator()
        
        final_metrics = {
            'total_time': total_time,
            'num_rounds': self.num_rounds,
            'num_clients': num_clients,
            'final_num_trees': self.history['num_trees'][-1] if self.history['num_trees'] else 0,
            'history': self.history,
        }
        
        if self.test_dataset is not None:
            test_metrics = self._evaluate_global(
                self.test_dataset.X, self.test_dataset.y, full_metrics=True)
            # Merge all unified metrics into final_metrics
            for k, v in test_metrics.items():
                if k != 'predictions':
                    final_metrics[k] = v
            # Backward compat aliases
            final_metrics['test_accuracy'] = test_metrics['accuracy']
            final_metrics['test_f1'] = test_metrics['f1_weighted']
            
            self._log(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
            self._log(f"Final Test F1w:      {test_metrics['f1_weighted']:.4f}")
            self._log(f"Final Cohen Kappa:   {test_metrics['cohen_kappa']:.4f}")
            self._log(f"Final MCC:           {test_metrics['mcc']:.4f}")
            if 'ece' in test_metrics:
                self._log(f"Final ECE:           {test_metrics['ece']:.4f}")
            self._log(f"Total Trees: {final_metrics['final_num_trees']}")
            self._log(f"Total Time: {total_time:.2f}s")
            self._log(f"Confusion Matrix:")
            for row in test_metrics['confusion_matrix']:
                self._log(f"  {row}")
        
        self._log_separator()
        
        return final_metrics
    
    def get_global_model(self) -> xgb.Booster | None:
        """Get the final global XGBoost model."""
        if self.global_model is None:
            return None
        
        params = self.xgb_params.copy()
        bst = xgb.Booster(params=params)
        bst.load_model(bytearray(self.global_model))
        return bst


def run_federated_xgboost_experiment(
    dataset,
    test_dataset=None,
    num_partitions=5,
    alpha=0.5,
    num_rounds=5,
    local_epochs=1,
    xgb_params=None,
    verbose=True,
):
    """Run a complete federated XGBoost experiment.
    
    Convenience function that creates partitioner and runs simulation.
    
    Parameters
    ----------
    dataset : TrainingDataset
        Full training dataset to partition.
    test_dataset : TrainingDataset, optional
        Held-out test set.
    num_partitions : int
        Number of FL clients. Default: 5
    alpha : float
        Dirichlet concentration for non-IID partitioning. Default: 0.5
    num_rounds : int
        Number of FL rounds. Default: 5
    local_epochs : int
        Local training epochs per round. Default: 1
    xgb_params : dict, optional
        XGBoost parameters.
    verbose : bool
        Enable verbose output. Default: True
    
    Returns
    -------
    dict
        Experiment results including metrics and history.
    """
    from utils import FederatedPartitioner
    
    if verbose:
        print(f"\n{'='*70}")
        print("EXPERIMENT C: Federated XGBoost with Dirichlet Partitioning")
        print(f"{'='*70}")
    
    # Create partitioner
    partitioner = FederatedPartitioner(
        dataset=dataset,
        num_partitions=num_partitions,
        alpha=alpha,
        seed=42,
    )
    
    if verbose:
        print(f"Created {num_partitions} partitions with alpha={alpha}")
        for i in range(num_partitions):
            part = partitioner.load_partition(i)
            unique, counts = np.unique(part.y, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"  Partition {i}: {len(part.X)} samples, distribution: {dist}")
    
    # Run simulation
    simulator = FedXGBoostSimulator(
        partitioner=partitioner,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        xgb_params=xgb_params,
        test_dataset=test_dataset,
        verbose=verbose,
    )
    
    results = simulator.run()
    results['partitioner'] = partitioner
    results['simulator'] = simulator
    
    return results


# =============================================================================
# FedAvg Simulator for MLP
# =============================================================================
class FedAvgMLPSimulator:
    """Simulates Federated Averaging with MLP models.

    Each client trains a local MLP; the server averages model weights.

    Parameters
    ----------
    partitioner : FederatedPartitioner
    num_rounds : int
    local_epochs : int
    hidden_dims : list of int
    dropout : float
    lr : float
    batch_size : int
    test_dataset : TrainingDataset, optional
    verbose : bool
    """

    def __init__(self, partitioner, num_rounds=5, local_epochs=2,
                 hidden_dims=None, dropout=0.3, lr=1e-3, batch_size=64,
                 test_dataset=None, verbose=True):
        import torch
        self.partitioner = partitioner
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.test_dataset = test_dataset
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_state = None
        self.history = {'round': [], 'test_accuracy': [], 'test_f1': [], 'round_time': []}

    def _log(self, msg):
        if self.verbose:
            print(f"[FL-Avg] {msg}")

    def _make_model(self, n_features, n_classes):
        from dl import MLP
        return MLP(n_features, self.hidden_dims, n_classes, dropout=self.dropout)

    def run(self):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from dl import MLP
        import copy

        num_clients = self.partitioner.num_partitions
        part0 = self.partitioner.load_partition(0)
        n_features = part0.X.shape[1]
        n_classes = len(self.partitioner.dataset.label_map)

        global_model = self._make_model(n_features, n_classes).to(self.device)
        self.global_state = copy.deepcopy(global_model.state_dict())

        self._log("=" * 70)
        self._log(f"Starting FedAvg MLP Simulation")
        self._log(f"Clients: {num_clients}, Rounds: {self.num_rounds}, Local epochs: {self.local_epochs}")
        self._log(f"Architecture: {self.hidden_dims}, LR: {self.lr}")
        self._log("-" * 70)

        total_start = time.time()

        for rnd in range(1, self.num_rounds + 1):
            rnd_start = time.time()
            self._log(f"\n{'='*20} ROUND {rnd}/{self.num_rounds} {'='*20}")

            client_states = []
            client_sizes = []

            for cid in range(num_clients):
                part = self.partitioner.load_partition(cid)
                X_c, y_c = part.X, part.y
                client_sizes.append(len(y_c))

                local_model = self._make_model(n_features, n_classes).to(self.device)
                local_model.load_state_dict(copy.deepcopy(self.global_state))

                ds = TensorDataset(torch.FloatTensor(X_c), torch.LongTensor(y_c))
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
                optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
                criterion = nn.CrossEntropyLoss()

                local_model.train()
                for _ in range(self.local_epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        optimizer.zero_grad()
                        loss = criterion(local_model(xb), yb)
                        loss.backward()
                        optimizer.step()

                client_states.append(copy.deepcopy(local_model.state_dict()))
                self._log(f"  Client {cid}: {len(y_c)} samples")

            # Weighted average
            total_samples = sum(client_sizes)
            avg_state = {}
            for key in self.global_state:
                avg_state[key] = sum(
                    client_states[i][key].float() * (client_sizes[i] / total_samples)
                    for i in range(num_clients)
                )
            self.global_state = avg_state

            # Evaluate
            test_acc, test_f1_val = 0.0, 0.0
            if self.test_dataset is not None:
                eval_model = self._make_model(n_features, n_classes).to(self.device)
                eval_model.load_state_dict(self.global_state)
                eval_model.eval()
                with torch.no_grad():
                    xte = torch.FloatTensor(self.test_dataset.X).to(self.device)
                    preds = eval_model(xte).argmax(dim=1).cpu().numpy()
                test_acc = accuracy_score(self.test_dataset.y, preds)
                test_f1_val = f1_score(self.test_dataset.y, preds, average='weighted', zero_division=0)
                self._log(f"  Test Accuracy: {test_acc:.4f}, F1: {test_f1_val:.4f}")

            rnd_time = time.time() - rnd_start
            self._log(f"  Round time: {rnd_time:.2f}s")
            self.history['round'].append(rnd)
            self.history['test_accuracy'].append(test_acc)
            self.history['test_f1'].append(test_f1_val)
            self.history['round_time'].append(rnd_time)

        total_time = time.time() - total_start

        # Final eval
        final = {'total_time': total_time, 'num_rounds': self.num_rounds,
                 'num_clients': num_clients, 'history': self.history}
        if self.test_dataset is not None:
            eval_model = self._make_model(n_features, n_classes).to(self.device)
            eval_model.load_state_dict(self.global_state)
            eval_model.eval()
            with torch.no_grad():
                xte = torch.FloatTensor(self.test_dataset.X).to(self.device)
                logits = eval_model(xte)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

            metrics = compute_all_metrics(
                self.test_dataset.y, preds, y_prob=probs, n_classes=n_classes)
            for k, v in metrics.items():
                final[k] = v
            # Backward compat aliases
            final['test_accuracy'] = metrics['accuracy']
            final['test_f1'] = metrics['f1_weighted']

            self._log("=" * 70)
            self._log("FINAL RESULTS")
            self._log(f"  Test Accuracy:     {metrics['accuracy']:.4f}")
            self._log(f"  Test F1 (weighted):{metrics['f1_weighted']:.4f}")
            self._log(f"  Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
            self._log(f"  MCC:               {metrics['mcc']:.4f}")
            if 'ece' in metrics:
                self._log(f"  ECE:               {metrics['ece']:.4f}")
            self._log(f"  Total Time:        {total_time:.2f}s")
            self._log(f"  Confusion Matrix:")
            for row in metrics['confusion_matrix']:
                self._log(f"    {row}")
            self._log("=" * 70)

        return final


if __name__ == '__main__':
    import sys, os, csv, argparse
    sys.path.insert(0, os.path.dirname(__file__))
    from utils import (load_all_datasets, load_all_datasets_cv,
                       FederatedPartitioner, TrainingDataset,
                       METRICS_CSV_FIELDS)

    parser = argparse.ArgumentParser(description='Federated Learning Experiments')
    parser.add_argument('--data-root', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', '..', 'wifi_sensing_data'),
                        help='Root folder containing dataset subfolders')
    parser.add_argument('--window', type=int, default=300, help='Window length')
    parser.add_argument('--sr', type=int, default=150, help='Guaranteed sample rate')
    parser.add_argument('--num-partitions', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1e6,
                        help='Dirichlet alpha (high=IID)')
    parser.add_argument('--num-rounds', type=int, default=20)
    parser.add_argument('--local-epochs', type=int, default=5)
    parser.add_argument('--cv', action='store_true',
                        help='Use temporal forward-chaining cross-validation')
    parser.add_argument('--n-folds', type=int, default=None,
                        help='Number of CV folds (auto if not set)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("FL EXPERIMENTS: FedXGBoost vs FedAvg  (unified metrics)")
    print("=" * 70)

    # Build dataset list: either fixed splits or temporal CV folds
    if args.cv:
        datasets_cv = load_all_datasets_cv(
            os.path.abspath(args.data_root),
            n_folds=args.n_folds,
            window_len=args.window,
            guaranteed_sr=args.sr,
            pipeline_name='rolling_variance',
            mode='flattened',
            var_window=20,
        )
        ds_fold_list = []
        for ds_name, folds in datasets_cv.items():
            for fold_idx, train_ds, test_ds in folds:
                ds_fold_list.append((ds_name, fold_idx, train_ds, test_ds))
    else:
        datasets = load_all_datasets(
            os.path.abspath(args.data_root),
            window_len=args.window,
            guaranteed_sr=args.sr,
            pipeline_name='rolling_variance',
            mode='flattened',
            var_window=20,
        )
        ds_fold_list = []
        for ds_name, (train_ds, test_ds) in datasets.items():
            ds_fold_list.append((ds_name, -1, train_ds, test_ds))

    xgb_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'max_depth': 4,
        'eta': 0.1,
        'num_parallel_tree': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    all_results = {}  # key -> metrics dict

    for ds_name, fold_idx, train_ds, test_ds in ds_fold_list:
        fold_tag = f"fold{fold_idx}" if fold_idx >= 0 else "fixed"
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  |  Split: {fold_tag}")
        print(f"  Train: {train_ds.X.shape}  Test: {test_ds.X.shape}")
        print(f"{'='*70}")

        if test_ds.X.shape[0] == 0:
            print(f"  SKIP - no test data")
            continue

        set_global_seed(42)

        partitioner = FederatedPartitioner(
            dataset=train_ds, num_partitions=args.num_partitions,
            alpha=args.alpha, seed=42)

        print(f"  Partitions ({args.num_partitions}, alpha={args.alpha}):")
        for i in range(args.num_partitions):
            p = partitioner.load_partition(i)
            unique, counts = np.unique(p.y, return_counts=True)
            print(f"    Client {i}: {len(p.X)} samples, "
                  f"dist={dict(zip(unique.tolist(), counts.tolist()))}")

        # --- FedXGBoost ---
        print(f"\n  --- FedXGBoost (Bagging) [{fold_tag}] ---")
        sim_xgb = FedXGBoostSimulator(
            partitioner=partitioner, num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            xgb_params=xgb_params, test_dataset=test_ds, verbose=args.verbose)
        xgb_res = sim_xgb.run()
        xgb_res['dataset'] = ds_name
        xgb_res['strategy'] = 'FedXGB'
        xgb_res['fold'] = fold_idx
        all_results[f"{ds_name}__{fold_tag}__FedXGB"] = xgb_res

        # --- FedAvg MLP ---
        print(f"\n  --- FedAvg (MLP) [{fold_tag}] ---")
        sim_avg = FedAvgMLPSimulator(
            partitioner=partitioner, num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            hidden_dims=[256, 128], dropout=0.3, lr=1e-3, batch_size=64,
            test_dataset=test_ds, verbose=args.verbose)
        avg_res = sim_avg.run()
        avg_res['dataset'] = ds_name
        avg_res['strategy'] = 'FedAvg'
        avg_res['fold'] = fold_idx
        all_results[f"{ds_name}__{fold_tag}__FedAvg"] = avg_res

    # ---- Final comparison (unified metrics) ----
    print(f"\n{'='*160}")
    print("FINAL FL COMPARISON: FedXGBoost vs FedAvg  (unified metrics)")
    print(f"{'='*160}")
    hdr = (f"{'Dataset':<25} {'Strategy':<10} | "
           f"{'Acc':>6} {'BalAcc':>6} {'F1w':>6} {'Kappa':>6} {'MCC':>6} "
           f"{'ECE':>6} | {'Time':>7}")
    print(hdr)
    print("-" * 120)
    for key in sorted(all_results.keys()):
        m = all_results[key]
        ece_val = m.get('ece', float('nan'))
        print(f"{m.get('dataset','?'):<25} {m.get('strategy','?'):<10} | "
              f"{m.get('accuracy',0):>6.4f} {m.get('balanced_accuracy',0):>6.4f} "
              f"{m.get('f1_weighted',0):>6.4f} {m.get('cohen_kappa',0):>6.4f} "
              f"{m.get('mcc',0):>6.4f} {ece_val:>6.4f} | "
              f"{m.get('total_time',0):>6.1f}s")

    # ---- Save results to CSV ----
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_tag = '_cv' if args.cv else ''
    csv_path = os.path.join(results_dir, f'fl_results{csv_tag}.csv')
    fieldnames = ['dataset', 'strategy', 'fold'] + METRICS_CSV_FIELDS + ['total_time']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for key in sorted(all_results.keys()):
            writer.writerow(all_results[key])
    print(f"\n[info] Results saved to {os.path.abspath(csv_path)}")

    print(f"\n{'='*70}")
    print("FL experiments completed!")
    print(f"{'='*70}")
