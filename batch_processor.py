'''
Utility functions for rtpose project
--------------------------------------------
Change to pytorch=0.4.0 by @LiMeng95
Utility functions for Multipose project
'''

import torch

def batch_processor(state, batch):
    gpus = state.params.gpus
    inp, heat_temp, heat_weight = batch

    if not state.model.training: # used for inference
        with torch.no_grad():
            input_var = inp.cuda(device=gpus[0])
            heat_weight_var = heat_weight.cuda(device=gpus[0], async=False)
            heat_temp_var = heat_temp.cuda(device=gpus[0], async=False)
    else:
        input_var = inp.cuda(device=gpus[0])
        heat_weight_var = heat_weight.cuda(device=gpus[0], async=False)
        heat_temp_var = heat_temp.cuda(device=gpus[0], async=False)

    inputs = [input_var]
    gts = [heat_temp_var, heat_weight_var, state.params.batch_size, gpus]
    saved_for_eval = []

    return inputs, gts, saved_for_eval
