import h5py
import numpy as np
import keras
import copy
import os
cwd = os.getcwd()

global_local_adr        =  os.path.dirname(os.path.abspath(__file__))+'/../'

def evaluate_emu_tf(emulator,_input):
    #To avoid the need to have install tensor flow
    import tensorflow as tf   
    @tf.function 
    def _evaluate_emu_tf(emulator,_input):
        return emulator(_input,training=False)  
    return _evaluate_emu_tf(emulator,_input)

class _GalaxyEmu():
    """
    Beta version of the galaxy clustering emulator
    """
    def __init__(self, network_adr = None, bounds = None, emulator_ID = 0, rmax = 85, extra = None):
        #emulator_ID: 0 -> wp; 1 -> xil0; 2 -> xil2; 3 -> xil4; 4 -> ds; 5 -> GAB; 6 -> kNN; 7 -> CIC; 8 -> VPF; 9 -> HOD; 10 -> CC_phi;
        emulator_name_arr = ['wp', 'xil0', 'xil2', 'xil4', 'ds', 'GAB', 'kNN', 'CIC', 'VPF', 'HOD', 'CC_phi'] #TODO extend
        emulator_name = emulator_name_arr[emulator_ID]
        if extra is not None: emulator_name += '_%s'%extra
        
        self.bounds         = bounds
        self.emulator_ID    = emulator_ID

        with h5py.File('%s/metadata_%s'%(network_adr,emulator_name), "r") as f:
            #self.mean, self.std = f['meandata'].attrs['i0'], f['meandata'].attrs['i1']
            self.mean, self.std = f['meandata'][0], f['meandata'][1]

        self.emulator       = keras.models.load_model('%s/%s.keras'%(network_adr,emulator_name))

        if emulator_ID < 4:
            rbins_wp_xil  = np.concatenate([np.logspace(np.log10(0.1),np.log10(77.9388), 16),np.linspace(85,140,12)])
            _r = np.concatenate([np.exp((np.log(rbins_wp_xil[:-1])+np.log(rbins_wp_xil[1:]))/2)[:16] , ((rbins_wp_xil[1:]+rbins_wp_xil[:-1])/2.)[-11:]])[:16]
            self.r = _r[_r<rmax]
        elif emulator_ID == 4:
            rbins_ds      = np.logspace(-1,1.6,14)
            self.r = np.exp((np.log(rbins_ds[1:]) + np.log(rbins_ds[:-1]))/2)


    def Norm(self, f, inverse = False):
        def _std(data, m=1):
            return np.nanstd(data[abs(data - np.nanmean(data)) < m * np.nanstd(data,axis=0)])
        if self.emulator_ID in [0,1]: #wp, xil0
            if self.emulator_ID == 0: _rfactor = 1
            else: _rfactor = 2 # wp is multiply by r, xil0 by r**2
            if inverse == False:
                _arr       = np.log(np.atleast_2d(f)*np.atleast_2d(self.r)**_rfactor)
                self.mean = np.nanmean(_arr)
                self.std  = _std(_arr)
                _arr       = (_arr-self.mean)/ self.std
            else:
                _arr = f*self.std+self.mean
                _arr = np.exp(np.atleast_2d(_arr))/np.atleast_2d(self.r)**_rfactor
            _arr[np.isnan(_arr)] = -3
            return _arr

        if self.emulator_ID in [2,3,4]: #xil2, xil4, ds
            if self.emulator_ID == 4: _rfactor, _mfactor = 1,1e12
            else: _rfactor, _mfactor = 1.5, 1
            if inverse == False:
                _arr       = np.atleast_2d(f)*np.atleast_2d(self.r)**_rfactor/_mfactor
                self.mean = np.nanmean(_arr)
                self.std  = _std(_arr)
                _arr       = (_arr-self.mean)/ self.std
            else:
                _arr = f*self.std+self.mean
                _arr = (np.atleast_2d(_arr))/np.atleast_2d(self.r)**_rfactor*_mfactor
            _arr[np.isnan(_arr)] = -3
            return _arr
        
        if self.emulator_ID in [5,6,7,8,9]: # gab, kNN, CIC, VPF, HOD
            if inverse == False:
                _arr       = np.atleast_2d(f)
                self.mean = np.nanmean(_arr)
                self.std  = _std(_arr)
                _arr       = (_arr-self.mean)/ self.std
            else:
                _arr = np.copy(f)*self.std+self.mean
                _arr = np.atleast_2d(_arr)
            _arr[np.isnan(_arr)] = -3
            return _arr    
            
        if self.emulator_ID in [10]: # CC_phi
            if inverse == False:
                _arr       = np.atleast_2d(np.log(f))
                self.mean = np.nanmean(_arr)
                self.std  = _std(_arr)
                _arr       = (_arr-self.mean)/ self.std
            else:
                _arr = np.copy(f)*self.std+self.mean
                _arr = np.atleast_2d(np.exp(_arr))
                _arr[np.isnan(_arr)] = 0
            return _arr
        
    def transform_space(self, x):
        return (x - self.bounds[:,0])/(self.bounds[:,1] - self.bounds[:,0])
        
    def get_clustering(self, params, TF_decorator = True, target_r = None):
        def ajust_r(self, out, target_r): #Not Tested
            if target_r is None: return out
            if ndim == 1:
                out2 =  np.interp(np.log10(target_r), np.log10(self.r), out) 
            else:
                out2 = np.zeros((len(out), len(target_r)))
                for j in range(len(out)):    
                    out2[j] = np.interp(np.log10(target_r), np.log10(self.r), out[j])
            return out2 

        # out = {}
        ndim = len(np.shape(params))
        if ndim == 1: _input = np.array([self.transform_space(params),])#np.atleast_2d()
        else: _input = self.transform_space(params)
        if TF_decorator:
            if ndim == 1: return ajust_r(self, self.Norm(evaluate_emu_tf(self.emulator,_input)[0], inverse=True)[0], target_r)
            else: return ajust_r(self, self.Norm(evaluate_emu_tf(self.emulator,_input), inverse=True), target_r)
        else:
            if ndim == 1: return ajust_r(self, self.Norm(self.emulator(_input,training=False)[0], inverse=True)[0], target_r)
            else: return ajust_r(self, self.Norm(self.emulator(_input,training=False), inverse=True), target_r)
    
    def get(self, params, TF_decorator = False, target_r = None, target_N = None, target_log_mh = None):
        def get_gab(x,a,b,c):
            return (c-1)*(scipy.special.erf((x-b)/a)+1)/2+1

        def get_kNN(x2, a, b, c, d):
            return a*(scipy.stats.skewnorm.sf(x2, d, loc = c, scale = b))

        def get_CIC(x, a, b, c):
            return (scipy.stats.skewnorm.cdf(x, c, loc = b, scale = a))

        def get_VPF(x2, b, c, d):
            return (scipy.stats.skewnorm.pdf(x2, d, loc = c, scale = b)/scipy.stats.skewnorm.pdf(0, d, loc = c, scale = b))

        def get_HOD_NCen_log(log_Mh, log_Mmin, sigmaLogM, sigmaLogM2, ampl ):
            #return np.log10(0.5 * (1 + scipy.special.erf((log_Mh - log_Mmin) / sigmaLogM)))
            return np.log10( ampl*skewnorm.cdf(log_Mh, sigmaLogM2, loc = log_Mmin, scale = sigmaLogM)                    )

        def get_HOD_NSat_log(log_Mh, log_M1, log_Mcut, alpha):
            return np.where(
                log_Mh < log_Mcut, -10, alpha * (np.log10(10**log_Mh - 10**log_Mcut) - log_M1)
            )

        if self.emulator_ID in [0, 1, 2, 3, 4]:
            return self.get_clustering(params, TF_decorator = TF_decorator , target_r = target_r)
        
        stat = self.get_clustering(params, TF_decorator = TF_decorator , target_r = None)
        if self.emulator_ID == 5: #5 -> GAB; 
            _r = np.log10(target_r)
            if len(np.shape(stat)) == 1:
                return get_gab(_r,stat[0],stat[1],stat[2])
            else:
                _out = np.empty((len(stat), len(_r)))
                for i in range(len(stat)):
                    _out[i] = get_gab(_r,stat[i][0],stat[i][1],stat[i][2])
            return _out
        if self.emulator_ID == 6: #6 -> kNN;
            _r = copy.deepcopy(target_r)
            if len(np.shape(stat)) == 1:
                return [get_kNN(_r, stat[0],stat[5],stat[10],stat[15]),
                        get_kNN(_r, stat[1],stat[6],stat[11],stat[16]),
                        get_kNN(_r, stat[2],stat[7],stat[12],stat[17]),
                        get_kNN(_r, stat[3],stat[8],stat[13],stat[18]),
                        get_kNN(_r, stat[4],stat[9],stat[14],stat[19])]
            else:
                _out = np.empty((len(stat), 5, len(_r)))
                for i in range(len(stat)):
                    _out[i] = [ get_kNN(_r, stat[i][0],stat[i][5],stat[i][10],stat[i][15]),
                                get_kNN(_r, stat[i][1],stat[i][6],stat[i][11],stat[i][16]),
                                get_kNN(_r, stat[i][2],stat[i][7],stat[i][12],stat[i][17]),
                                get_kNN(_r, stat[i][3],stat[i][8],stat[i][13],stat[i][18]),
                                get_kNN(_r, stat[i][4],stat[i][9],stat[i][14],stat[i][19])]
            return _out
        
        if self.emulator_ID == 7: #7 -> CIC;
            _N = copy.deepcopy(target_N)
            if len(np.shape(stat)) == 1:
                return [get_CIC(_N, stat[0],stat[5],stat[10]),
                        get_CIC(_N, stat[1],stat[6],stat[11]),
                        get_CIC(_N, stat[2],stat[7],stat[12]),
                        get_CIC(_N, stat[3],stat[8],stat[13]),
                        get_CIC(_N, stat[4],stat[9],stat[14])]
            else:
                _out = np.empty((len(stat), 5, len(_N)))
                for i in range(len(stat)):
                    _out[i] = [ get_CIC(_N, stat[i][0],stat[i][5],stat[i][10]),
                                get_CIC(_N, stat[i][1],stat[i][6],stat[i][11]),
                                get_CIC(_N, stat[i][2],stat[i][7],stat[i][12]),
                                get_CIC(_N, stat[i][3],stat[i][8],stat[i][13]),
                                get_CIC(_N, stat[i][4],stat[i][9],stat[i][14])]
            return _out
        
        if self.emulator_ID == 8: #8 -> VPF;
            _r = copy.deepcopy(target_r)
            if len(np.shape(stat)) == 1:
                return get_VPF(_r, stat[0],stat[1],stat[2])
            else:
                _out = np.empty((len(stat), len(_r)))
                for i in range(len(stat)):
                    _out[i] = get_VPF(_r, stat[i][0],stat[i][1],stat[i][2])
            return _out

        if self.emulator_ID == 9: #9 -> HOD;
            _logMh = copy.deepcopy(target_log_mh)
            if len(np.shape(stat)) == 1:
                Ncen = get_HOD_NCen_log(_logMh, stat[0],stat[1],stat[2],stat[3])
                Nsat = get_HOD_NSat_log(_logMh, stat[4],stat[5],stat[6])
                return Ncen, Nsat, np.log10(10**Ncen+10**Nsat)
            else:
                _out = np.empty((len(stat), 3, len(_logMh)))
                for i in range(len(stat)):
                    Ncen = get_HOD_NCen_log(_logMh, stat[i][0],stat[i][1],stat[i][2],stat[i][3])
                    Nsat = get_HOD_NSat_log(_logMh, stat[i][4],stat[i][5],stat[i][6])
                    Ntot = np.log10(10**Ncen+10**Nsat)
                    _out[i] = Ncen, Nsat, Ntot
            return _out

        if self.emulator_ID == 10: #10 -> CC_phi;
            return stat


def GalaxyEmu( emulator_ID = None, emulator_Name = None, emulator_version = 0, network_adr = None, bounds = None,extra = None, rmax = 85):#
    if network_adr is None: network_adr = global_local_adr
    network_adr += '/v%d/'%emulator_version

    if bounds is None:
        bounds = np.array([
                   [0,1.8],         # sigma log(M)
                   [-1.5,1.2],      # log(tmerger)
                   [-0.5,0.5],      # fkP
                   [-0.5,0.5],      # fkM
                   [0,1],           # betta
                   [0.65,0.9],      # Sigma8 | Extended Lv 2
                   [0.23,0.40],     # OmegaM | Extended Lv 1
                   [0.04,0.06],     # Omegab | Fixed Planck
                   [0.92,1.01],     # ns     | Fixed Planck
                   [0.6,0.8],       # h      | Extended Lv 2
                   [0,0.4],         # Mnu
                   [-4, -2.25],     # log(n)
                   [0,2]])          # z

    if ((emulator_Name is not None) and (emulator_ID is not None) or ((emulator_Name is None) and (emulator_ID is None)) ):
        print('GalaxyEmu need "emulator_ID" or "emulator_Name" Code Terminated', emulator_ID, emulator_Name)
        return None
        
    if emulator_ID is None:
        emulator_ID = np.where(emulator_Name.lower() == np.array(['wp', 'xil0', 'xil2', 'xil4', 'ds', 'gab', 'knn', 'cic', 'hod']))[0][0]

    if (extra is None) and (emulator_ID in [0,4]): extra = '30Mpc'

    return _GalaxyEmu(network_adr = network_adr, bounds = bounds, emulator_ID = emulator_ID, rmax = rmax, extra = extra)
    
