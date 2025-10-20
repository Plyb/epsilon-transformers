import collections
import joblib
import torch
import torch.nn.functional as F

nested_dict_factory = collections.defaultdict

def load(path):
    checkpoint_path = path
    checkpoint = joblib.load(checkpoint_path)
    layer = checkpoint['combined']
    basis = layer['basis']
    rmse = layer['rmse']
    return {
        'basis': torch.from_numpy(basis),
        'rmse': rmse
    }

mess3 = load('belief_regression_results/20251018175842_mess3_0/checkpoint_15513600.joblib')
bloch = load('belief_regression_results/20251018175842_bloch_0/checkpoint_15513600.joblib')

print(f'rmse mess3: {mess3['rmse']}, bloch: {bloch['rmse']}')

similarities = F.cosine_similarity(mess3['basis'].T.unsqueeze(1), bloch['basis'].T.unsqueeze(0), dim=-1)
print(similarities)

