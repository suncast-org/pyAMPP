#!/usr/bin/env python3

import ctypes
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import astropy.units as u
from astropy.time import Time
import os

class GXRadioImageComputing:
    def __init__(self, libname):
        self.libname = libname
        libc_mw=ctypes.CDLL(self.libname)
        self.mwfunc=libc_mw.pyComputeMW
        self.mwfunc.restype=ctypes.c_int

    def load_ebtel(self, ebtel_file):
        ebtel_data = io.readsav(ebtel_file)
        s = ebtel_data["lrun"].shape
    
        ebtel_dtypes = [("DEM_on", np.int32),
                        ("DDM_on", np.int32),
                        ("NQ", np.int32),
                        ("NL", np.int32),
                        ("NT", np.int32),
                        ("Qrun", np.float32, ebtel_data["qrun"].shape),
                        ("Lrun", np.float32, ebtel_data["lrun"].shape),
                        ("logtdem", np.float32, ebtel_data["logtdem"].size)
                      ]
        if "dem_cor_run" in ebtel_data.keys():
            ebtel_dtypes.append(("DEM_cor_run", np.float32, ebtel_data["dem_cor_run"].shape))
        
        if "ddm_cor_run" in ebtel_data.keys():
            ebtel_dtypes.append(("DDM_cor_run", np.float32, ebtel_data["ddm_cor_run"].shape))
        
        ebtel_dt = np.dtype(ebtel_dtypes)
        ebtel_c = np.zeros(1, dtype=ebtel_dt)
        
        ebtel_c["DEM_on"] = 1 if "dem_cor_run" in ebtel_data.keys() else 0
        ebtel_c["DDM_on"] = 1 if "ddm_cor_run" in ebtel_data.keys() else 0
        ebtel_c["NQ"] = s[1]
        ebtel_c["NL"] = s[0]
        ebtel_c["NT"] = ebtel_data["logtdem"].size
        ebtel_c["Qrun"] = np.float32(ebtel_data["qrun"])
        ebtel_c["Lrun"] = np.float32(ebtel_data["lrun"])
        ebtel_c["logtdem"] = np.float32(ebtel_data["logtdem"])
        
        if "dem_cor_run" in ebtel_data.keys():
            ebtel_c["DEM_cor_run"] = np.float32(ebtel_data["DEM_cor_run"])
        
        if "ddm_cor_run" in ebtel_data.keys():
            ebtel_c["DDM_cor_run"] = np.float32(ebtel_data["DDM_cor_run"])
    
        return ebtel_c, ebtel_dt
    
    def load_model_dict(self, model_dict, header):
        lon = header["CRVAL1"]
        lat = header["CRVAL2"]
        obs_time = Time(header["DATE_OBS"])
        dsun_obs = header["DSUN_OBS"]
    
        init_coords = SkyCoord(lon*u.deg, lat*u.deg, frame=frames.HeliographicCarrington,rsun=696000*u.km,\
                           obstime=obs_time, observer="earth")
        hcc = init_coords.transform_to(frames.Heliocentric)
        obstime = obs_time.unix - 283996800 # according to IDL specs, anytim function
    
        x, y, z = hcc.cartesian.x, hcc.cartesian.y, hcc.cartesian.z
        xC, yC, zC = (c.to(u.cm).value for c in (x, y, z))
        # Warning: the values xC, yC, zC slightly differ, because sunpy Heliocentric system is different from IDL
    
        DSun = dsun_obs*1e2
        RSun = init_coords.rsun.to(u.cm).value
        
        lonC=np.rad2deg(np.arctan2(xC, zC))
        latC=np.rad2deg(np.arcsin(yC/RSun))
    
        dr = model_dict["dr"]
    
        dx=dr[0]*RSun
        dy=dr[1]*RSun
        dz_uniform=dr[2]*RSun
        dz=model_dict["dz"].T*RSun

        s = dz.shape
        sc = model_dict["bcube"].T.shape
        
        Nx=s[2]
        Ny=s[1]
        Nz=s[0]
    
        chromo_layers=model_dict["chromo_layers"]
        corona_layers=Nz-chromo_layers
        corona_base=model_dict["corona_base"]
    
        Bx=np.zeros(s, dtype=np.float32, order="C")
        By=np.zeros(s, dtype=np.float32, order="C")
        Bz=np.zeros(s, dtype=np.float32, order="C")
    
        chromo_bcube = model_dict["chromo_bcube"].T
        bcube = model_dict["bcube"].T
    
        Bx[0:chromo_layers, :, :]=chromo_bcube[0, :, :, :]
        By[0:chromo_layers, :, :]=chromo_bcube[1, :, :, :]
        Bz[0:chromo_layers, :, :]=chromo_bcube[2, :, :, :]
    
        Bx[chromo_layers : Nz, :, :]=bcube[0, corona_base : Nz, :, :]
        By[chromo_layers : Nz, :, :]=bcube[1, corona_base : Nz, :, :]
        Bz[chromo_layers : Nz, :, :]=bcube[2, corona_base : Nz, :, :]
    
        chr_s  = (chromo_layers,  Ny, Nx)
        cor_s  = (sc[1]-corona_base, Ny, Nx)
        cor_s2 = (corona_base,    Ny, Nx)
    
        chromo_n0 =np.zeros(chr_s)
        chromo_np =np.zeros(chr_s)
        chromo_nHI=np.zeros(chr_s)
        chromo_T0 =np.zeros(chr_s)
    
        chromo_idx = model_dict["chromo_idx"]
        chromo_n0.flat[chromo_idx] = model_dict["chromo_n"]
        chromo_np.flat[chromo_idx] = model_dict["n_p"]
        chromo_nHI.flat[chromo_idx]= model_dict["n_hi"]
        chromo_T0.flat[chromo_idx] = model_dict["chromo_t"]

        QB = model_dict["avfield"].copy()
        QL = model_dict["physlength"].copy()*RSun
        uu = model_dict["status"]
        QB[(uu & 4) != 4] = 0
        QL[(uu & 4) != 4] = 0

        corona_Bavg         = QB[:, :, corona_base : sc[1]].T
        chromo_uniform_Bavg = QB[:, :, 0 : corona_base].T

        corona_L         = QL[:, :, corona_base : sc[1]].T
        chromo_uniform_L = QL[:, :, 0 : corona_base].T

        model_dt_varlist = [
                ('Nx', np.int32),
                ('Ny', np.int32),
                ('Nz', np.int32),
                ('chromo_layers', np.int32),
                ('corona_layers', np.int32),
                ('corona_base',   np.int32),
                ('DSun', np.float64),
                ('RSun', np.float64),
                ('lonC', np.float64),
                ('latC', np.float64),
                ('dx',   np.float64),
                ('dy',   np.float64),
                ('dz_uniform', np.float64),
                ('obstime',    np.float64),
                ('dz', np.float32, s),
                ('Bx', np.float32, s),
                ('By', np.float32, s),
                ('Bz', np.float32, s),
                ('chromo_n0' ,  np.float32, chr_s),
                ('chromo_np' ,  np.float32, chr_s),
                ('chromo_nHI',  np.float32, chr_s),
                ('chromo_T0',   np.float32, chr_s),
                ('corona_Bavg', np.float32, cor_s),
                ('corona_L',    np.float32, cor_s),
                ('chromo_uniform_Bavg', np.float32, cor_s2),
                ('chromo_uniform_L',    np.float32, cor_s2)]
        
        model_dt = np.dtype(model_dt_varlist)
        model = np.zeros(1, dtype=model_dt)
    
        for vars in model_dt_varlist:
            k, dt = vars[0], vars[1]
            val = dt(locals()[k])
            if len(val.shape) > 1:
                model[k] = val.astype(dt, order="C")
            else:
                model[k] = val
    
        return model, model_dt
    
    def load_model_sav(self, file_name):
        model_data = io.readsav(file_name)
        lon = model_data.box.index[0].CRVAL1[0]
        lat = model_data.box.index[0].CRVAL2[0]
    
        init_coords = SkyCoord(lon*u.deg, lat*u.deg, frame=frames.HeliographicCarrington,rsun=696000*u.km,\
                           obstime=Time(model_data.box.index[0]["DATE_OBS"][0]), observer="earth")
        hcc = init_coords.transform_to(frames.Heliocentric)
        aptime = Time(model_data.box.index[0]["DATE_OBS"][0])
        obstime = aptime.unix - 283996800 # according to IDL specs, anytim function
    
        x, y, z = hcc.cartesian.x, hcc.cartesian.y, hcc.cartesian.z
        xC, yC, zC = (c.to(u.cm).value for c in (x, y, z))
        # Warning: the values xC, yC, zC slightly differ, because sunpy Heliocentric system is different from IDL
    
        DSun = model_data.box.index[0]["DSUN_OBS"][0]*1e2
        RSun = init_coords.rsun.to(u.cm).value
        
        lonC=np.rad2deg(np.arctan2(xC, zC))
        latC=np.rad2deg(np.arcsin(yC/RSun))
    
        dr = model_data.box.dr[0]
    
        dx=dr[0]*RSun
        dy=dr[1]*RSun
        dz_uniform=dr[2]*RSun
        dz=model_data.box.dz[0]*RSun
    
        s = dz.shape
        sc = model_data.box.bcube[0].shape
        
        Nx=s[2]
        Ny=s[1]
        Nz=s[0]
    
        chromo_layers=model_data.box.chromo_layers[0]
        corona_layers=Nz-chromo_layers
        corona_base=model_data.box.corona_base[0]
    
        Bx=np.zeros(s, dtype=np.float32, order="C")
        By=np.zeros(s, dtype=np.float32, order="C")
        Bz=np.zeros(s, dtype=np.float32, order="C")
    
        chromo_bcube = model_data.box.chromo_bcube[0]
        bcube = model_data.box.bcube[0]
    
        Bx[0:chromo_layers, :, :]=chromo_bcube[0, :, :, :]
        By[0:chromo_layers, :, :]=chromo_bcube[1, :, :, :]
        Bz[0:chromo_layers, :, :]=chromo_bcube[2, :, :, :]
    
        Bx[chromo_layers : Nz, :, :]=bcube[0, corona_base : Nz, :, :]
        By[chromo_layers : Nz, :, :]=bcube[1, corona_base : Nz, :, :]
        Bz[chromo_layers : Nz, :, :]=bcube[2, corona_base : Nz, :, :]
    
        chr_s  = (chromo_layers,  Ny, Nx)
        cor_s  = (sc[1]-corona_base, Ny, Nx)
        cor_s2 = (corona_base,    Ny, Nx)
    
        chromo_n0 =np.zeros(chr_s)
        chromo_np =np.zeros(chr_s)
        chromo_nHI=np.zeros(chr_s)
        chromo_T0 =np.zeros(chr_s)
    
        chromo_idx = model_data.box.chromo_idx[0]
        chromo_n0.flat[chromo_idx] = model_data.box.chromo_n[0].flat
        chromo_np.flat[chromo_idx] = model_data.box.n_p[0].flat
        chromo_nHI.flat[chromo_idx]= model_data.box.n_hi[0].flat
        chromo_T0.flat[chromo_idx] = model_data.box.chromo_t[0].flat
    
        if "BMED" in model_data.box.dtype.names:
            QB = np.zeros((Nx, Ny, sc[1]))
            QL = np.zeros((Nx, Ny, sc[1]))
            idx = np.unravel_index(model_data.box.idx[0], QB.shape, order="F")
    
            QB[idx] = model_data.box.bmed[0]
            QL[idx] = model_data.box.length[0]*RSun
        elif "AVFIELD" in model_data.box.dtype.names:
            QB = model_data.box.avfield[0].T.copy()
            QL = model_data.box.physlength[0].T.copy()*RSun
            uu = model_data.box.status[0].T
            QB[(uu & 4) != 4] = 0
            QL[(uu & 4) != 4] = 0
    
        corona_Bavg         = QB[:, :, corona_base : sc[1]].T
        chromo_uniform_Bavg = QB[:, :, 0 : corona_base].T
    
        corona_L         = QL[:, :, corona_base : sc[1]].T
        chromo_uniform_L = QL[:, :, 0 : corona_base].T
    
        model_dt_varlist = [
                ('Nx', np.int32),
                ('Ny', np.int32),
                ('Nz', np.int32),
                ('chromo_layers', np.int32),
                ('corona_layers', np.int32),
                ('corona_base',   np.int32),
                ('DSun', np.float64),
                ('RSun', np.float64),
                ('lonC', np.float64),
                ('latC', np.float64),
                ('dx',   np.float64),
                ('dy',   np.float64),
                ('dz_uniform', np.float64),
                ('obstime',    np.float64),
                ('dz', np.float32, s),
                ('Bx', np.float32, s),
                ('By', np.float32, s),
                ('Bz', np.float32, s),
                ('chromo_n0' ,  np.float32, chr_s),
                ('chromo_np' ,  np.float32, chr_s),
                ('chromo_nHI',  np.float32, chr_s),
                ('chromo_T0',   np.float32, chr_s),
                ('corona_Bavg', np.float32, cor_s),
                ('corona_L',    np.float32, cor_s),
                ('chromo_uniform_Bavg', np.float32, cor_s2),
                ('chromo_uniform_L',    np.float32, cor_s2)]
        
        model_dt = np.dtype(model_dt_varlist)
        model = np.zeros(1, dtype=model_dt)
    
        for vars in model_dt_varlist:
            k, dt = vars[0], vars[1]
            #print(k)
            val = dt(locals()[k])
            if len(val.shape) > 1:
                model[k] = val.astype(dt, order="C")
            else:
                model[k] = val
    
        return model, model_dt

    def synth_model(self, model, model_dt, ebtel, ebtel_dt, freqlist, box_Nx, box_Ny, box_xc, box_yc, box_dx, box_dy, Tbase, nbase, Q0, a, b, force_isothermal=1):
        dt_s=np.dtype([('Nx', np.int32),
                       ('Ny', np.int32),
                       ('Nf', np.int32),
                       ('_r1', np.int32),
                       ('xc', np.float64),
                       ('yc', np.float64),
                       ('dx', np.float64),
                       ('dy', np.float64),
                       ('freqlist', np.float64, len(freqlist))])
        simbox=np.zeros(1, dtype=dt_s)
        simbox['Nx']=box_Nx
        simbox['Ny']=box_Ny
        simbox['Nf']=len(freqlist)
        simbox['xc']=box_xc
        simbox['yc']=box_yc
        simbox['dx']=box_dx
        simbox['dy']=box_dy
        simbox['freqlist']=freqlist
        
        dt_c=np.dtype([('Tbase', np.float64),
                       ('nbase', np.float64),
                       ('Q0', np.float64),
                       ('a', np.float64),
                       ('b', np.float64),
                       ('iso', np.int32)])
        cparms=np.zeros(1, dtype=dt_c)
        cparms['Tbase']=Tbase
        cparms['nbase']=nbase
        cparms['Q0']=Q0
        cparms['a']=a
        cparms['b']=b
        cparms['iso']=force_isothermal

        dt_o=np.dtype([('flagsAll', np.int32, 6),
                       ('flagsCorona', np.int32, 6),
                       ('TI', np.float64, (box_Nx, box_Ny, len(freqlist))),
                       ('TV', np.float64, (box_Nx, box_Ny, len(freqlist)))])
        outspace=np.zeros(1, dtype=dt_o)

        _dt_s=np.ctypeslib.ndpointer(dtype=dt_s)
        _dt_c=np.ctypeslib.ndpointer(dtype=dt_c)
        _dt_o=np.ctypeslib.ndpointer(dtype=dt_o)
        _dt_ebtel=np.ctypeslib.ndpointer(dtype=ebtel_dt)
        _dt_model=np.ctypeslib.ndpointer(dtype=model_dt)

        self.mwfunc.argtypes=[_dt_model, _dt_ebtel, _dt_s, _dt_c, _dt_o]

        r=self.mwfunc(model, ebtel, simbox, cparms, outspace)

        o=outspace[0]
        TI=np.reshape(np.ravel(o['TI']), (box_Nx, box_Ny, len(freqlist)), order='F')
        TV=np.reshape(np.ravel(o['TV']), (box_Nx, box_Ny, len(freqlist)), order='F')
    
        return {"TI": TI, "TV": TV}
    