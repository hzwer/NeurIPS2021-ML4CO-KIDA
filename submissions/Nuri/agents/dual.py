import os
import torch
import ecole as ec
import numpy as np
import collections
import random
from agents.agent_model import GNNPolicyItem, GNNPolicyLoad, GNNPolicyAno


class ObservationFunction(ec.observation.NodeBipartite):

    def __init__(self, problem):
        super().__init__()

    def seed(self, seed):
        pass


class Policy():

    def __init__(self, problem):
        self.rng = np.random.RandomState()

        self.device = f"cuda:0"
        self.problem = problem
        
        if problem == 'item_placement':
            params_path = 'item.pkl'
            self.policy = GNNPolicyItem().to(self.device)
        elif problem == 'load_balancing':
            #params_path = 'load.pkl'
            self.policy = None
        elif problem == 'anonymous':
            params_path = 'ano.pkl' 
            self.policy = GNNPolicyAno().to(self.device)
        else:
            params_path = 'item.pkl'
            self.policy = GNNPolicyItem().to(self.device)
        if problem == 'item_placement':
            policy0 = GNNPolicyItem().to(self.device)
            policy1 = GNNPolicyItem().to(self.device)
            policy2 = GNNPolicyItem().to(self.device)
            policy0.load_state_dict(torch.load('item0.pkl'))
            policy1.load_state_dict(torch.load('item1.pkl'))
            policy2.load_state_dict(torch.load('item2.pkl'))
            models = [policy0,policy1,policy2]
            worker_state_dict=[x.state_dict() for x in models]
            weight_keys=list(worker_state_dict[0].keys())
            fed_state_dict=collections.OrderedDict()
            for key in weight_keys:
                key_sum=0
                for i in range(len(models)):
                    key_sum=key_sum+worker_state_dict[i][key]
                fed_state_dict[key]=key_sum/len(models)
            self.policy.load_state_dict(fed_state_dict)
            self.policy.eval()
        elif problem == 'load_balancing':
            continue
        else:
            self.policy.load_state_dict(torch.load(params_path))
            self.policy.eval()

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        if self.problem == 'load_balancing':
            return random.choice(action_set)
        else:
            variable_features = observation.column_features
            variable_features = np.delete(variable_features, 14, axis=1)
            variable_features = np.delete(variable_features, 13, axis=1)

            constraint_features = torch.FloatTensor(observation.row_features).to(self.device)
            edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64)).to(self.device)
            edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1)).to(self.device)
            variable_features = torch.FloatTensor(variable_features).to(self.device)
            action_set = torch.LongTensor(np.array(action_set, dtype=np.int64)).to(self.device)

            logits = self.policy(constraint_features, edge_index, edge_attr, variable_features)
            logits = logits[action_set]
            action_idx = logits.argmax().item()
            action = action_set[action_idx]

            return action
