
import torch

def get_loss(pb_type):
    if pb_type in ['maxcut', 'vqls', 'vqls_direct']:
        from .objective.hamiltonian import Energy
        return Energy.apply
