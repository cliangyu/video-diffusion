import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np
import os

BASE_DIR = '/ubc/cs/research/plai-scratch/wsgh/video-diffusion/'



import wandb


class IdentityDict(dict):
    def __init__(self, d):
        for k, v in d.items():
            self[k] = v
    def __getitem__(self, k):
        if k in self:
            return super().__getitem__(k)
        else:
            return k

nice_mode_names = IdentityDict({
    'autoreg': 'Autoregressive',
    'adaptive-autoreg': 'Adaptive autoregressive',
    'adaptive-hierarchy-2': 'Adaptive hierarchy-2',
    'independent': 'Only original observations',
    'mixed-autoreg-independent': 'Mixed',
    'hierarchy-2': 'Hierarchy-2',
    'hierarchy-3': 'Hierarchy-3',
    'hierarchy-4': 'Hierarchy-4',
    'hierarchy-5': 'Hierarchy-5',
    'really-independent': 'Independent',
    'cwvae': 'CWVAE',
    'differently-spaced-groups': 'Training distribution',
    'hierarchy-2_optimal-linspace-t-force-nearby': 'Optimal hierarchy-2',
    'autoreg_optimal-linspace-t-force-nearby': 'Optimal autoreg',
})


def parse_run(path):
    try:
        relative_path = Path(str(path).replace(BASE_DIR, ""))
        results, *subdirs, wandb_id, ckpt_name, inference_mode, last_bit = relative_path.parts
        assert results == 'results'
        *mode, max_frames, step_size, T, obs_length = inference_mode.split('_')
        mode = '_'.join(mode)
        max_frames, step_size, T, obs_length = map(eval, [max_frames, step_size, T, obs_length])
        mode = nice_mode_names[mode]
    except Exception as e:
        print(path)
        raise e
    return dict(subdirs=subdirs, wandb_id=wandb_id, ckpt_name=ckpt_name,
                mode=mode, max_frames=max_frames, step_size=step_size, T=T, obs_length=obs_length, last_bit=last_bit)

def pad_to_len(s, l):
    return s + ' '*max(0, l-len(s))

def directory(path):
    return path if path.is_dir() else path.parent

def first_n(array, n):
    assert len(array) >= n
    return array[:n]

def format_mean_std(mean, std, n=None):
    s = f"{mean:.2f} +- {std:.3f}"
    if n is not None:
        s += f" ({n} seeds)"
    return s


class DB:
    def __init__(self, data=[]):
        if len(data) == 0:
            self.data = []
            for path in self.get_dirs():
                self.add_from_path(path)
        else:
            self.data = data
            
    def add_from_path(self, path):
        path_data = {"path": path, "config": parse_run(path), **self.get_data(path)}
        if 'no_include' not in path_data:
            self.data.append(path_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def query(self, **kwargs):
        return ExpDB([x for x in self.data if all([x["config"][k] == v for (k,v) in kwargs.items()])])

    def sort(self, keys):
        if isinstance(keys, str):
            self.sort([keys])
        self.data.sort(key=lambda x: [x["config"][k] for k in keys])
        
    def get_sorted(self, keys):
        if isinstance(keys, str):
            return self.get_sorted([keys])
        return ExpDB(data=sorted(self.data, key=lambda x: [x["config"][k] for k in keys]))

    def print_config(self):
        for x in self.data:
            print(x["config"])
            
    def get_data(self, path):
        raise NotImplementedError


class ElboDB(DB):
    ext = ""
    
    def get_dirs(self):
        paths = Path(BASE_DIR).glob('results/**/elbos/')
        return [p for p in paths]
    
    def get_data(self, path):
        i = 0
        while os.path.exists(path / f"elbo_{i}{self.ext}.pkl"):
            data_i = pickle.load(open(path / f"elbo_{i}{self.ext}.pkl", 'rb'))
            if i == 0:
                data = {k: [] for k in data_i}
            for k in data:
                data[k].append(data_i[k])
            i += 1
        if i == 0:
            print('WARNING: failed', str(path))
            raise Exception
        return {'data': {k: np.stack(v, axis=0) for k, v in data.items()}}


class MetricDB(DB):

    def get_dirs(self):
        paths = Path(BASE_DIR).glob('results/**/metrics_*.pkl')  
        return [p for p in paths]

    def get_data(self, path):
        fname = path.parts[-1]
        n_videos, n_samples, video_length = fname.split('_')[1].split('.')[0].split('-')
        if int(n_videos) < 100:
            return {'no_include': 0}
        return {'data': pickle.load(open(path, "rb")), 'n_videos': n_videos,
                'n_samples': n_samples, 'video_length': video_length}


class RespaceMetricDB(MetricDB):

    def get_dirs(self):
        paths = Path(BASE_DIR).glob('results/*/*/*respace*/*/metrics_*.pkl')  
        print(paths)
        return [p for p in paths]

class RespaceElboDB(ElboDB):
    ext = "_respace250"

    def get_dirs(self):
        paths = Path(BASE_DIR).glob('results/*/*/ema*500000*respace250/*/elbos')  
        print(paths)
        return [p for p in paths]




class IDHasRPENet(dict):
    def __init__(self):
        self.api = wandb.Api()
        
    def __getitem__(self, id):
        if id in self:
            return super().__getitem__(id)
        else:
            self[id] = self.api.run(f"universal-conditional-ddpm/video-diffusion/{id}").config['use_rpe_net']
            return self[id]
id_to_rpe_net = IDHasRPENet()




respace_elbo_db = RespaceElboDB()
#for run in respace_elbo_db:
#    print(run['config']['subdirs'])
#    print('rpe net', id_to_rpe_net[run['config']['wandb_id']])
#    print(first_n(run['data']['mse'], 100).shape)

datasets = ['mazes',]# 'minerl']

ablation_ys = {
    dataset+'-'+ablation: ([], [])
    for dataset in datasets
    for ablation in ['rpe', 'no_rpe', 'uniform']
}

xs = [1, 19, 49, 99, 149, 199, 249, 299]
for dataset in datasets:
    dataset_runs = [run for run in respace_elbo_db if dataset in ''.join(run['config']['subdirs'])]
    for x in xs:
        x_runs = [run for run in dataset_runs if f'0-{x}-2' in run['config']['mode']]
        uniform_runs = [run for run in x_runs if 'uniform' in ''.join(run['config']['subdirs'])]
        rpe = [run for run in x_runs if 'rpe-net' in ''.join(run['config']['subdirs']) and id_to_rpe_net[run['config']['wandb_id']]]
        no_rpe = [run for run in x_runs if 'rpe-net' in ''.join(run['config']['subdirs']) and not id_to_rpe_net[run['config']['wandb_id']]]
        print(dataset, x, len(dataset_runs), len(x_runs), len(uniform_runs), len(rpe), len(no_rpe))
        for ablation, ab_runs in zip(['rpe', 'no_rpe', 'uniform'], [rpe, no_rpe, uniform_runs]):
            wandb_ids = set(run['config']['wandb_id'] for run in ab_runs)
            print(ablation, wandb_ids)
            mis = []
            for wandb_id in wandb_ids:
                run = [run for run in ab_runs if run['config']['wandb_id'] == wandb_id and 'independent' not in str(run['path'])][0]
                ind_run = [run for run in ab_runs if run['config']['wandb_id'] == wandb_id and 'independent' in str(run['path'])][0]
                mi = (ind_run['data']['mse'].sum(axis=1) - run['data']['mse'].sum(axis=1)).mean(axis=0)
                mis.append(mi)
            mis = np.array(mis)
            mean_mi = mis.mean()
            std_mi = mis.std() if len(mis) > 1 else 0
            ablation_ys[dataset+'-'+ablation][0].append(mean_mi)
            ablation_ys[dataset+'-'+ablation][1].append(std_mi)

for dataset in ['minerl', 'mazes']:
    fig, ax = plt.subplots()
    for name, (ys, stds) in ablation_ys.items():
        if dataset not in name:
            continue
        ys = np.array(ys)
        stds = np.array(stds)
        print(stds)
        ax.plot(xs, ys, label=name)
        ax.fill_between(xs, ys-stds, ys+stds, color='k', alpha=0.2)
    ax.legend()
    ax.set_ylim(0)
    fig.savefig(f'rpe-net-effect-vs-distance-{dataset}.pdf', bbox_inches='tight')
                


respace_metric_db = RespaceMetricDB()

def print_mode_fvds(db):
    to_show = {}
    for run in db:
        metrics = run['data']['fvd']
        inf_mode = run['config']['mode']
        string = f"{format_mean_std(metrics.mean(), metrics.std()/np.sqrt(len(metrics)-1), len(metrics))} - {inf_mode}"
        to_show[string] = metrics.mean()
    for label, elbo in sorted(to_show.items(), key=lambda x: x[1]):
        print(label)

#print(list(respace_metric_db))
#print([r['config'] for r in respace_metric_db])
#for r in respace_metric_db:
#    config = r['config']
##    print()
#    print(r['config']['subdirs'], r['config']['wandb_id'], config['mode'], config['T'], config['last_bit'])
#    print_mode_fvds([r])

for s in ['mazes-uniform', 'mazes-rpe', 'minerl-uniform', 'minerl-rpe']:
    print(s, '+rpe')
    print_mode_fvds([r for r in respace_metric_db if s in ''.join(r['config']['subdirs']) and id_to_rpe_net[r['config']['wandb_id']]])
    print(s)
    print_mode_fvds([r for r in respace_metric_db if s in ''.join(r['config']['subdirs']) and not id_to_rpe_net[r['config']['wandb_id']]])







#elbo_db = ElboDB()
#metric_db = MetricDB()

if False:

    minerl_ablation_elbo_db = [i for i in elbo_db if 'minerl-rpe-net-ablation' in i['config']['subdirs']]
    mazes_ablation_elbo_db = [i for i in elbo_db if 'mazes-rpe-net-ablation' in i['config']['subdirs']]

    def get_elbo_mean_std(db, has_rpe_net=None, metric='total_bpd', T=None):
        relevant_data = []
        for run in db:
            if has_rpe_net is None or id_to_rpe_net[run['config']['wandb_id']] == has_rpe_net:
                data = first_n(run['data'][metric], n=100).sum(axis=1).mean(axis=0)
                relevant_data.append(data)
        relevant_data = np.array(relevant_data)
        if T is not None:
            relevant_data = relevant_data / (T-36)
        assert relevant_data.ndim == 1
        return relevant_data.mean(), relevant_data.std(), len(relevant_data)

    minerl_training_elbo_db = [i for i in minerl_ablation_elbo_db if i['config']['mode'] == 'Training distribution']
    mazes_training_elbo_db = [i for i in mazes_ablation_elbo_db if i['config']['mode'] == 'Training distribution']
    print('\nMINERL ELBOs\n')
    print('With RPE net')
    print(format_mean_std(*get_elbo_mean_std(minerl_training_elbo_db, has_rpe_net=True)))
    print('Without RPE net')
    print(format_mean_std(*get_elbo_mean_std(minerl_training_elbo_db, has_rpe_net=False)))

    print('\nMAZES ELBOs\n')
    print('With RPE net')
    print(format_mean_std(*get_elbo_mean_std(mazes_training_elbo_db, has_rpe_net=True)))
    print('Without RPE net')
    print(format_mean_std(*get_elbo_mean_std(mazes_training_elbo_db, has_rpe_net=False)))


# make RPE net ablation
if False:

    def get_spacing_i_db(db, i, independent, with_rpe_net):
        mask_name = f"linspace-no-obs-0-{i}-2"
        if independent:
            mask_name = 'independent-'+mask_name
        return [run for run in db if run['config']['mode'] == mask_name if id_to_rpe_net[run['config']['wandb_id']] == with_rpe_net]

    def plot(ax, x, runs, color, label, diff_to=None):
        values = np.array([first_n(run['data']['total_bpd'], 100).sum(axis=1).mean(axis=0) for run in runs])
        if diff_to is not None:
            diff_to_values = np.array([first_n(run['data']['total_bpd'], 100).sum(axis=1).mean(axis=0) for run in diff_to])
            values = diff_to_values - values
        ax.errorbar(np.array(x)+1, y=values.mean(), yerr=values.std(), color=color, label=label, fmt='o')
    
        fig, axes = plt.subplots(ncols=2)
        for ax, db in zip(axes, [minerl_ablation_elbo_db, mazes_ablation_elbo_db]):
            xs = [1, 19, 49, 99, 149, 199, 249, 299, 349, 399, 449, 499]  # 149, 249
            for x in xs:
                with_runs = get_spacing_i_db(db, x, independent=False, with_rpe_net=True)
                without_runs = get_spacing_i_db(db, x, independent=False, with_rpe_net=False)
                plot(ax, x, with_runs, color='b', label='With RPE network' if x==xs[0] else None)
                plot(ax, x, without_runs, color='r', label='Without RPE network' if x==xs[0] else None)

            ax.set_xlabel('Index of second frame')
            ax.set_ylabel('NLL per pixel')
        fig.savefig('../syncing-plots/rpe-net-effect-vs-distance.pdf', bbox_inches='tight')

    minerl_inf_mode_db = [i for i in metric_db if 'minerl-final-850k' in i['config']['subdirs']]
    minerl_inf_mode_elbo_db = [i for i in elbo_db if 'minerl-final-850k' in i['config']['subdirs']]

    carla_inf_mode_db = [i for i in metric_db if 'carla-final' in i['config']['subdirs']]

    mazes_inf_mode_db = [i for i in metric_db if 'mazes-final-950k' in i['config']['subdirs']]
    mazes_inf_mode_elbo_db = [i for i in elbo_db if 'mazes-final-950k' in i['config']['subdirs']]

    mazes_autoreg_db = [i for i in metric_db if 'mazes-autoreg-600k' in i['config']['subdirs']]
    minerl_autoreg_db = [i for i in metric_db if 'minerl-autoreg-550k' in i['config']['subdirs']]


    def print_mode_elbos(db, metric='mse'):
        for run in db:
            #metrics = run['data'][metric]
            inf_mode = run['config']['mode']
            print(format_mean_std(*get_elbo_mean_std([run], metric='mse', T=300)), inf_mode)

    print('\n\nMineRL ELBOs')
    print_mode_elbos(minerl_inf_mode_elbo_db)
    print('\n\nMazes ELBOs')
    print_mode_elbos(mazes_inf_mode_elbo_db)


    def print_mode_fvds(db):
        to_show = {}
        for run in db:
            metrics = run['data']['fvd']
            inf_mode = run['config']['mode']
            string = f"{format_mean_std(metrics.mean(), metrics.std()/np.sqrt(len(metrics)-1), len(metrics))} - {inf_mode}"
            to_show[string] = metrics.mean()
        for label, elbo in sorted(to_show.items(), key=lambda x: x[1]):
            print(label)
    
    print('Mazes')
    print_mode_fvds(mazes_inf_mode_db)

    print('MineRL')
    print_mode_fvds(minerl_inf_mode_db)

    print('CARLA')
    print_mode_fvds(carla_inf_mode_db)

    print('Now some FVDs for models traing on Autoreg')
    print('Mazes')
    print_mode_fvds(mazes_autoreg_db)
    print('MineRL')
    print_mode_fvds(minerl_autoreg_db)


if False:
    # print miscellaneos elbos from Saeid
    minerl_google_fs4 = [run for run in elbo_db if 'minerl-google-fs4' in str(run['config']['subdirs'])]
    mazes_google_fs4 = [run for run in elbo_db if 'mazes-google-fs4' in str(run['config']['subdirs'])]
    mazes_opt_hierarchy = [run for run in elbo_db if 'mazes-final-950k' in str(run['config']['subdirs']) and run['config']['mode'] == 'Optimal hierarchy-2']
    mazes_opt_autoreg = [run for run in elbo_db if 'mazes-final-950k' in str(run['config']['subdirs']) and run['config']['mode'] == 'Optimal autoreg']
    minerl_opt_hierarchy = [run for run in elbo_db if 'minerl-final-850k' in str(run['config']['subdirs']) and run['config']['mode'] == 'Optimal hierarchy-2']
    minerl_opt_autoreg = [run for run in elbo_db if 'minerl-final-850k' in str(run['config']['subdirs']) and run['config']['mode'] == 'Optimal autoreg']

    print(format_mean_std(*get_elbo_mean_std(mazes_google_fs4, metric='mse', T=300)), 'Google thing on Mazes')
    print(format_mean_std(*get_elbo_mean_std(minerl_google_fs4, metric='mse', T=500)), 'Google thing on MineRL')
    print(format_mean_std(*get_elbo_mean_std(mazes_opt_hierarchy, metric='mse', T=300)), 'Opt hierarchy on Mazes')
    print(format_mean_std(*get_elbo_mean_std(mazes_opt_autoreg, metric='mse', T=300)), 'Opt autoreg on Mazes')
    print(format_mean_std(*get_elbo_mean_std(minerl_opt_hierarchy, metric='mse', T=500)), 'Opt hierarchy on MineRL')
    print(format_mean_std(*get_elbo_mean_std(minerl_opt_autoreg, metric='mse', T=500)), 'Opt autoreg on MineRL')

    # and not from Saied:
    minerl_autoreg = [run for run in elbo_db if 'minerl-autoreg-550k' in str(run['config']['subdirs'])]
    mazes_autoreg = [run for run in elbo_db if 'mazes-autoreg-600k' in str(run['config']['subdirs'])]
    print(format_mean_std(*get_elbo_mean_std(minerl_autoreg, metric='mse', T=500)), 'Autoreg on MineRL')
    print(format_mean_std(*get_elbo_mean_std(mazes_autoreg, metric='mse', T=300)), 'Autoreg on Mazes')
