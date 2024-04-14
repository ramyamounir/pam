

def get_pam_configs():

    transition_configs = {
            'synaptic_density': 1.0,
            'eta_inc': 0.1,
            'eta_dec': 0.0,
            'eta_decay': 0.0,
            'init_mean': 0.0,
            'init_std': 0.1,
            'threshold': 0.8,
            }

    emission_configs = {
            'synaptic_density': 1.0,
            'eta_inc': 0.1,
            'eta_dec': 0.1,
            'eta_decay': 0.0,
            'init_mean': 0.0,
            'init_std': 0.1,
            'threshold': 0.1,
            }

    return dict(transition_configs=transition_configs, emission_configs=emission_configs)


def get_pc_configs(l=1):

    if l==1:
        configs = {
                'lr': 1e-4 ,
                'data_type': 'binary',
                'learn_iters': 800,
                }
    else:
        configs = {
                'learn_lr': 1e-4,
                'inf_lr': 1e-2,
                'learn_iters': 800,
                'inf_iters': 400,
                }

    return configs


def get_hn_configs(d=1):

    configs = {
            'data_type': 'binary',
            'sep': d,
            }
    return configs



