import seaborn as sns

method_names = {
    'PAM-1': 'PAM $N_k=1$', 
    'PAM-4': 'PAM $N_k=4$', 
    'PAM-8': 'PAM $N_k=8$', 
    'PAM-16': 'PAM $N_k=16$', 
    'PAM-24': 'PAM $N_k=24$', 
    'PC-1': 'tPC', 
    'PC-2': 'tPC $L=2$', 
    'HN-1-5': 'AHN d=1', 
    'HN-1-50': 'AHN d=1, W=$0.5N_c$',
    'HN-2-5': 'AHN d=2', 
    'HN-2-50': 'AHN d=2, W=$0.5N_c$',
        }

method_ids = { v: i for i, (k, v) in enumerate(method_names.items())}
method_colors = {v:c for (k, v), c in zip(method_names.items(), sns.color_palette(n_colors=len(method_names)))}

