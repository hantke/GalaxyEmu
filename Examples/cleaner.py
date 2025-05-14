import numpy as np
import deepdish as dd
import copy
import time

def get_localtime():
    t = time.localtime()
    return time.strftime("[%D-%H:%M:%S]", t)

def clean_bounds(bounds, fix_prop):
    for i in range(len(fix_prop)):
        bounds = np.delete(bounds,fix_prop[-i-1][0],axis = 0)
    return bounds

raw_bounds = np.array([
                   [0,1.8],         # sigma log(M)
                   [-1.5,1.2],      # log(tmerger)
                   [-1.5,1.5],      # fkP
                   [-1.5,1.5],      # fkM
                   [0,1],           # betta
                   [0.65,0.9],      # Sigma8 | Extended Lv 2
                   [0.23,0.40],     # OmegaM | Extended Lv 1
                   [0.04,0.06],     # Omegab | Fixed Planck
                   [0.92,1.01],     # ns     | Fixed Planck
                   [0.6,0.8],       # h      | Extended Lv 2
                   [0,0.4],         # Mnu
                   [-4, -2.25],     # log(n)
                   [0,2]])          # z


def clean(Arr):

    _target_vector        = []
    _index                = []
    Arr['Clean']['r_arr'] = []
    
    for i in Arr['Raw']['stats']:
        _r = Arr['Raw']['r_arr'][i]
        rmin = Arr['Raw']['rmin_wp'] if i != 4 else Arr['Raw']['rmin_ds']
        rmax = Arr['Raw']['rmax_wp'] if i != 4 else Arr['Raw']['rmax_ds']
        mask = (Arr['Raw']['r_arr'][i] > rmin) & (Arr['Raw']['r_arr'][i] < rmax)
        Arr['Clean']['r_arr'].append(Arr['Raw']['r_arr'][i][mask])
        _target_vector = np.concatenate([_target_vector, Arr['Raw']['target_cluster'][i][mask]])
        _index = np.concatenate([_index, np.where(mask)[0]+i*len(Arr['Raw']['r_arr'][0])]) #CAREFULL, ONLY WORKS IF rw_p == r_xil
    _index = _index.astype(np.int32)

    Arr['Clean']['target_vector'] = _target_vector

    Nvector = len(_target_vector)
    Arr['Clean']['Cv'] = np.zeros((Nvector,Nvector))
    for i in range(Nvector):
        for j in range(Nvector):
            Arr['Clean']['Cv'][i][j] = Arr['Raw']['Cv'][_index[i]][_index[j]]
    if Arr['Raw']['err_diag'] is not None:
        Arr['Clean']['Cv'] += np.diag((Arr['Raw']['err_diag']*Arr['Clean']['target_vector'])**2)
    if Arr['Raw']['diag']:
        Arr['Clean']['Cv'] += np.diag(np.diag(Arr['Clean']['Cv']))

    str_stat = ''
    for i in range(len(Arr['Raw']['stats'])):
        str_stat+= str(Arr['Raw']['stats'][i])
        if i != len(Arr['Raw']['stats'])-1: str_stat += '_'

    if Arr['Raw']['output_name'] is None:
        extra = ''
        if Arr['Raw']['err_diag'] is not None:
            extra += '_%.02f'%Arr['Raw']['err_diag']
        Arr['Clean']['output_name'] = 'Model_stat_%s_rmin_wp_%.01f_rmax_wp_%.01f_rmin_ds_%.01f_rmax_ds_%.01f_diag_%s%s.hdf5'%(str_stat, Arr['Raw']['rmin_wp'], Arr['Raw']['rmax_wp'], Arr['Raw']['rmin_ds'], Arr['Raw']['rmax_ds'], Arr['Raw']['diag'], extra)
    else:
        Arr['Clean']['output_name'] = copy.deepcopy(Arr['Raw']['output_name'])

    Arr['Status'] = 'Clean'
    Arr['Clean']['stats']            = copy.deepcopy(Arr['Raw']['stats'])
    Arr['Clean']['bounds']           = clean_bounds(raw_bounds, Arr['Raw']['fix_prop'])  
    Arr['Clean']['fix_prop']         = copy.deepcopy(Arr['Raw']['fix_prop'])
    Arr['Clean']['nprint']           = 10
    Arr['Clean']['npoints']          = 20000
    Arr['Clean']['niter']            = 400
    Arr['Clean']['backup_frequency'] = 1000000 #Not really necesary for a quick PSO
    Arr['Clean']['c1']               = 0.4
    Arr['Clean']['c2']               = 0.2
    Arr['Clean']['w']                = 0.95
    Arr['Clean']['reflect_param']    = 0.95
    Arr['Clean']['verbose']          = True

    Arr['Clean']['MCMC_npoints']     = 1000
    Arr['Clean']['MCMC_nsteps']      = 5000
    return Arr
