'''
Utility functions for rtpose project
--------------------------------------------
Change to pytorch=0.4.0 by @LiMeng95
Utility functions for Multipose project
'''

import torch

def batch_processor(state, batch):
    gpus = state.params.gpus
    subnet_name = state.params.subnet_name  # 'detection_subnet'/'keypoint_subnet'/'prn_subnet'

    if subnet_name == 'keypoint_subnet':
        inp, heat_temp, heat_weight = batch

        if not state.model.training:  # used for inference
            with torch.no_grad():
                input_var = inp.cuda(device=gpus[0])
                heat_weight_var = heat_weight.cuda(device=gpus[0], async=False)
                heat_temp_var = heat_temp.cuda(device=gpus[0], async=False)
        else:
            input_var = inp.cuda(device=gpus[0])
            heat_weight_var = heat_weight.cuda(device=gpus[0], async=False)
            heat_temp_var = heat_temp.cuda(device=gpus[0], async=False)

        inputs = [[input_var, subnet_name]]
        gts = [subnet_name, heat_temp_var, heat_weight_var]
        saved_for_eval = []
    elif subnet_name == 'detection_subnet':  #'detection_subnet'
        inp, anno = batch  # anno: [x1, y1, x2, y2, category_id]

        if not state.model.training:  # used for inference
            with torch.no_grad():
                input_var = inp.cuda(device=gpus[0])
                anno_var = anno.cuda(device=gpus[0])
        else:
            input_var = inp.cuda(device=gpus[0])
            anno_var = anno.cuda(device=gpus[0])

        inputs = [[input_var, subnet_name]]
        gts = [subnet_name, anno_var]
        saved_for_eval = []
    else:  #'prn_subnet'
        inp, label = batch  # input, label

        if not state.model.training:  # used for inference
            with torch.no_grad():
                input_var = inp.cuda(device=gpus[0]).float()
                anno_var = label.cuda(device=gpus[0]).float()
        else:
            input_var = inp.cuda(device=gpus[0]).float()
            anno_var = label.cuda(device=gpus[0]).float()

        inputs = [[input_var, subnet_name]]
        gts = [subnet_name, anno_var]
        saved_for_eval = []

    return inputs, gts, saved_for_eval

