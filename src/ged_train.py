#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ************************************
# @Time     : 2019/3/3 22:34
# @Author   : Xiang Ling
# @Lab      : nesa.zju.edu.cn
# @File     : ged_train.py
# ************************************
import numpy as np
import os
import torch
import torch.nn.functional as functional
from datetime import datetime

from data import GEDDataset
from ged_config import ged_args
from model.DenseGraphMatching import MultiLevelGraphMatchNetwork
from simgnn_utils import computing_precision_ks
from utils import create_dir_if_not_exists, write_log_file
from utils import metrics_kendall_tau, metrics_spearmanr_rho, metrics_mean_square_error


class GEDTrainer(object):
    
    def __init__(self, data_dir, device, best_model_path, args, log_path):
        super(GEDTrainer, self).__init__()
        
        # training parameters
        self.max_iterations = args.iterations
        self.iter_val_start = args.iter_val_start
        self.iter_val_every = args.iter_val_every
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.device = device
        self.validation_data = None
        
        self.best_model_path = best_model_path
        self.log_file = log_path
        
        self.dataset = GEDDataset(ged_main_dir=data_dir, args=args)
        self.flag_inclusive = args.inclusive
        write_log_file(self.log_file, str(args))
        
        self.model = MultiLevelGraphMatchNetwork(node_init_dims=self.dataset.input_dim, arguments=args, device=self.device).to(self.device)
        write_log_file(self.log_file, str(self.model))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
        print("\n\n", self.model.state_dict().keys())
    
    def _need_val(self, iteration):
        return iteration >= self.iter_val_start and iteration % self.iter_val_every == 0
    
    def batch_pairs_predication(self, batch_feature_1, batch_adjacent_1, batch_mask_1, batch_feature_2, batch_adjacent_2, batch_mask_2):
        feature_1 = np.array(batch_feature_1)
        feature_2 = np.array(batch_feature_2)
        adj_1 = np.array(batch_adjacent_1)
        adj_2 = np.array(batch_adjacent_2)
        predictions = self.model(batch_x_p=feature_1, batch_x_h=feature_2, batch_adj_p=adj_1, batch_adj_h=adj_2)
        return predictions
    
    def training_batch_predication(self, batch_feature_1, batch_adjacent_1, batch_mask_1, batch_feature_2, batch_adjacent_2, batch_mask_2, ged_pairs):
        
        self.model.train()
        self.optimizer.zero_grad()
        predictions = self.batch_pairs_predication(batch_feature_1, batch_adjacent_1, batch_mask_1, batch_feature_2, batch_adjacent_2, batch_mask_2)
        trues = torch.from_numpy(np.array(ged_pairs, dtype=np.float32)).to(self.device)
        
        loss = functional.mse_loss(predictions, trues)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), torch.stack((trues, predictions), 1)
    
    def val_batch_predication(self, batch_feature_1, batch_adjacent_1, batch_mask_1, batch_feature_2, batch_adjacent_2, batch_mask_2, ged_pairs):
        st_time = datetime.now()
        self.model.eval()
        nr_examples = batch_adjacent_1.shape[0]  # Number of graphs
        assert batch_feature_1.shape[0] == batch_adjacent_1.shape[0] and batch_feature_2.shape[0] == batch_adjacent_2.shape[0]
        st = 0
        batch_size = self.batch_size
        predictions = []
        with torch.no_grad():
            while st < nr_examples:
                if st + batch_size >= nr_examples:
                    ed = nr_examples
                else:
                    ed = st + batch_size
                feature_1 = batch_feature_1[st:ed]
                feature_2 = batch_feature_2[st:ed]
                adjacent_1 = batch_adjacent_1[st:ed]
                adjacent_2 = batch_adjacent_2[st:ed]
                mask_1 = batch_mask_1[st:ed]
                mask_2 = batch_mask_2[st:ed]
                batch_pred = self.batch_pairs_predication(feature_1, adjacent_1, mask_1, feature_2, adjacent_2, mask_2)
                predictions.append(batch_pred)
                st = ed
        
        predictions = torch.cat(predictions)
        
        trues = torch.from_numpy(np.array(ged_pairs, dtype=np.float32)).to(self.device)
        loss = torch.nn.functional.mse_loss(predictions, trues)
        return loss.data.item(), np.stack((trues.cpu().detach().numpy(), predictions.cpu().detach().numpy()), 1), datetime.now() - st_time
    
    def testing_prediction(self):
        results = np.zeros((len(self.dataset.testing_graphs), len(self.dataset.train_val_graphs)))
        write_log_file(self.log_file, 'result shape is {} '.format(results.shape))
        for row in range(len(self.dataset.testing_graphs)):
            batch_rows_feature, batch_rows_adjacent, batch_rows_mask, batch_cols_feature, batch_cols_adjacent, batch_cols_mask = self.dataset.extract_test_matrices(row)
            st = 0
            pred = []
            while st < len(self.dataset.train_val_graphs):
                if st + self.batch_size < len(self.dataset.train_val_graphs):
                    ed = st + self.batch_size
                else:
                    ed = len(self.dataset.train_val_graphs)
                batch_rows_feature_small = batch_rows_feature[st:ed]
                batch_rows_adjacent_small = batch_rows_adjacent[st:ed]
                batch_rows_mask_small = batch_rows_mask[st:ed]
                batch_cols_feature_small = batch_cols_feature[st:ed]
                batch_cols_adjacent_small = batch_cols_adjacent[st:ed]
                batch_cols_mask_small = batch_cols_mask[st:ed]
                
                with torch.no_grad():
                    cur_pred = self.batch_pairs_predication(batch_rows_feature_small, batch_rows_adjacent_small, batch_rows_mask_small, batch_cols_feature_small,
                                                            batch_cols_adjacent_small, batch_cols_mask_small)
                    pred.append(cur_pred)
                st = ed
            pred = torch.cat(pred)
            results[row] = pred.detach().cpu().numpy()
        
        return results
    
    def fit(self):
        self.model.train()
        
        time = datetime.now()
        best_val_loss = None
        for iteration in range(self.max_iterations):
            batch_feature_1, batch_adj_1, batch_mask_1, batch_feature_2, batch_adj_2, batch_mask_2, batch_ged = self.dataset.get_training_batch()
            train_loss, train_true_pred = self.training_batch_predication(batch_feature_1, batch_adj_1, batch_mask_1, batch_feature_2, batch_adj_2, batch_mask_2, batch_ged)
            
            # print in training steps
            if iteration % int(self.max_iterations / 20) == 0:
                time_spent = datetime.now() - time
                time = datetime.now()
                write_log_file(self.log_file, "Iteration = {}\tbatch loss={} (e-3) @ {}".format(iteration, train_loss * 1000, time_spent))
            
            # validation
            if self._need_val(iteration=iteration):
                self.model.eval()
                # only load once at first
                if self.validation_data is None:
                    self.validation_data = self.dataset.get_all_validation()
                    val_feature_1, val_adj_1, val_mask_1, val_feature_2, val_adj_2, val_mask_2, val_ged = self.validation_data
                else:
                    val_feature_1, val_adj_1, val_mask_1, val_feature_2, val_adj_2, val_mask_2, val_ged = self.validation_data
                
                val_loss, val_true_pred, time_spent = self.val_batch_predication(val_feature_1, val_adj_1, val_mask_1, val_feature_2, val_adj_2, val_mask_2, val_ged)
                
                write_log_file(self.log_file, "\nvalidation iteration={}, loss={}(e-3), spend time = {} @ {}".format(iteration, val_loss * 1000, time_spent, datetime.now()))
                if not best_val_loss or val_loss <= best_val_loss:
                    write_log_file(self.log_file, '\tvalidation mse decreased ( {} ---> {} (e-3) ), and save the model ... '.format(best_val_loss, val_loss * 1000))
                    best_val_loss = val_loss
                    
                    torch.save(self.model.state_dict(), self.best_model_path)
                
                write_log_file(self.log_file, '\tbest validation mse = {} (e-3)'.format(best_val_loss * 1000))
    
    def testing(self):
        # load the last checkpoint with the best model
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        self.model.to(self.device)
        
        # Double check validation
        if self.validation_data is None:
            self.validation_data = self.dataset.get_all_validation()
            val_feature_1, val_adj_1, val_mask_1, val_feature_2, val_adj_2, val_mask_2, val_ged = self.validation_data
        else:
            val_feature_1, val_adj_1, val_mask_1, val_feature_2, val_adj_2, val_mask_2, val_ged = self.validation_data
        
        val_loss, val_true_pred, time_spent = self.val_batch_predication(val_feature_1, val_adj_1, val_mask_1, val_feature_2, val_adj_2, val_mask_2, val_ged)
        write_log_file(self.log_file, "\nDouble check validation, loss = {}(e-3) @ {}".format(val_loss * 1000, datetime.now()))
        
        # testing
        test_predictions = self.testing_prediction()
        test_mse = metrics_mean_square_error(self.dataset.ground_truth.flatten(), test_predictions.flatten())
        test_rho = metrics_spearmanr_rho(self.dataset.ground_truth.flatten(), test_predictions.flatten())
        test_tau = metrics_kendall_tau(self.dataset.ground_truth.flatten(), test_predictions.flatten())
        ps, inclusive_true_ks, inclusive_pred_ks = computing_precision_ks(trues=self.dataset.ground_truth, predictions=test_predictions, ks=[10, 20], inclusive=self.flag_inclusive, rm=0)
        test_results = {
            'mse': test_mse,
            'rho': test_rho,
            'tau': test_tau,
            'test_p10': ps[0],
            'test_p20': ps[1]
        }
        write_log_file(self.log_file, 'Test results:')
        for k, v in test_results.items():
            write_log_file(self.log_file, '\t {} = {}'.format(k, v))


if __name__ == '__main__':
    d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = ged_args.gpu_index
    
    create_dir_if_not_exists(ged_args.log_path)
    log_root_dir = ged_args.log_path
    signature = ged_args.dataset + '@' + datetime.now().strftime("%Y-%m-%d@%H:%M:%S")
    current_run_dir = os.path.join(log_root_dir, signature)
    create_dir_if_not_exists(current_run_dir)
    model_save_path = os.path.join(current_run_dir, 'best_model.pt')
    log_file_path = os.path.join(current_run_dir, 'log.txt')
    ged_main_dir = ged_args.data_dir
    

    trainer = GEDTrainer(data_dir=ged_main_dir, device=d, best_model_path=model_save_path, args=ged_args, log_path=log_file_path)
    trainer.fit()
    trainer.testing()
