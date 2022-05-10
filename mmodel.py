import json
import numpy as np
import os
from scipy import sparse
import pandas as pd
from time import perf_counter
import swiglpk

#Constant as defined in Compass manuscript
BETA = 0.95


mmodels_dir = 'Metabolic Models/'
class MModel:
    def __init__(self, ub, lb, rxns, mets, S, penalties, cache, unidir=False):
        self.ub= ub
        self.lb = lb
        self.rxns = rxns
        self.mets = mets 
        self.S = S
        self.penalties = penalties
        self.cache = cache
        self.unidir = unidir
        self.partner_rxns = {}
        
    def get_penalties_vector(self, sample_index):
        return self.penalties[:,sample_index]

def read_penalties(model_name):
    path = os.path.join('compass_runs', model_name, '_tmp/penalties.txt.gz')
    return pd.read_csv(path, sep="\t", header=0, index_col=0)

def read_cache(model_name):
    path = os.path.join('cache', model_name, 'default-media', 'preprocess.json')
    with open(path, 'r') as f:
        cache = json.load(f)
        f.close()
    return cache

def read_model(model_name):
    model_path = os.path.join(mmodels_dir, model_name, 'model')
    with open(os.path.join(model_path, 'model.ub.json'), 'r') as f:
        ub = json.load(f)
        f.close()
    with open(os.path.join(model_path, 'model.lb.json'), 'r') as f:
        lb = json.load(f)
        f.close()
    with open(os.path.join(model_path, 'model.rxns.json'), 'r') as f:
        rxns = json.load(f)
        f.close()
    with open(os.path.join(model_path, 'model.mets.json'), 'r') as f:
        mets = json.load(f)
        f.close()
    with open(os.path.join(model_path, 'model.S.json'), 'r') as f:
        S = sijv_to_coo(np.array(json.load(f)).T)
        f.close()
    
    #Note that penalties for _pos and _neg are the same
    penalties = read_penalties(model_name).rename(index=lambda x: x[:-4])
    penalties = penalties.loc[penalties.index.intersection(rxns)].to_numpy()
    
    cache = read_cache(model_name)
    
    model = MModel(ub=ub, lb=lb, rxns=rxns, mets=mets, S=S, penalties=penalties, cache=cache)
    return model

def sijv_to_coo(S_ijv):
    return sparse.coo_array((S_ijv[2], (S_ijv[0].astype('int64')-1, S_ijv[1].astype('int64')-1)))

def flip_rxn_dir(rxn):
    if rxn.endswith('_pos'):
        return rxn[:-4] + '_neg'
    elif rxn.endswith('_neg'):
        return rxn[:-4] + '_pos'
    else:
        print("Error")
        
def make_unidirectional(model):
    assert (model.unidir == False)
    
    S2, ub2, lb2, rxns2, penalties2 = [], [], [], [], []
    S = model.S.tocsc()
    
    partners = {}
    for i in range(len(model.rxns)):
        if model.ub[i] > 0:
            rxns2.append(model.rxns[i] + '_pos')
            ub2.append(model.ub[i])
            lb2.append(0)
            S2.append(S[:,[i]])
            penalties2.append(i)
        if model.lb[i] < 0:
            rxns2.append(model.rxns[i] + '_neg')
            ub2.append(-model.lb[i])
            lb2.append(0)
            S2.append(-S[:,[i]])
            penalties2.append(i)
        if model.ub[i] > 0 and model.lb[i] < 0:
            partners[model.rxns[i] + '_pos'] = model.rxns[i] + '_neg'
            partners[model.rxns[i] + '_neg'] = model.rxns[i] + '_pos'
            
    penalties2 = model.penalties[penalties2,:]
            
    model_uni = MModel(ub=ub2, lb=lb2, rxns=rxns2, mets=model.mets, 
                       S=sparse.hstack(S2).T.tocoo(), penalties=penalties2, cache=model.cache, 
                       unidir=True)
    model_uni.partner_rxns = partners
    return model_uni