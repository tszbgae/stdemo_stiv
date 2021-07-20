# -*- coding: utf-8 -*-
"""
Spyder Editor

this script transforms the by hour npys into monthly npys in ../stivnpys/transformed

then transforms into by month npy of shape [15,mondays,ys,xs] onto usb drive folder stivnpys
"""
import pygrib
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np

lmon=[31,28,31,30,31,30,31,30,30,31,30,31]

st4ll=Dataset('/media/ats/Backup/stiv/ST4.2005050102.01h.nc')
lats=st4ll.variables['latitude'][:]
lons=st4ll.variables['longitude'][:]

a = pygrib.open('/media/ats/Backup/stiv/2017/ST4.2017050100.01h')
grb = a.select(name='Total Precipitation')[0]
bs=grb.values

latss,latsn,lonsw,lonse=400,600,700,950

def parser(inarr):
    return inarr[latss:latsn,lonsw:lonse]
def stringer(innum):
    if innum<10:
        outs='0'+str(innum)
    else:
        outs=str(innum)
    return outs

#transform by hour npys to by month summary npys
for yr in range(2002,2017):
    if os.path.exists('/home/ats/stivnpys/transformed/'+str(yr))==False:
        os.mkdir('/home/ats/stivnpys/transformed/'+str(yr))
    for mon in range(5,9):
        aout=np.zeros((lmon[mon-1],24,200,250))
        for d in range(1,lmon[mon-1]+1):
            sd=stringer(d)
            for hr in range(0,24):
                sh=stringer(hr)
                try:

                    a=np.load('/home/ats/stivnpys/'+str(yr)+'/'+str(mon)+'/'+str(yr)+'0'+str(mon)+sd+sh+'.npy')
                    
                except:
                    print('FAIL, '+str(yr)+'0'+str(mon)+sd+sh)
                    a=a0
                aout[d-1,hr,:,:]=a
                a0=a
            print(str(yr)+'0'+str(mon)+sd)
        np.save('/home/ats/stivnpys/transformed/'+str(yr)+'/'+str(mon)+'.npy',aout)
#%%

for mon in range(5,9):
    aout=np.zeros((15,lmon[mon-1],24,200,250))
    for yr in range(2002,2017):
        a=np.load('/home/ats/stivnpys/transformed/'+str(yr)+'/'+str(mon)+'.npy')
        aout[yr-2002,:,:,:,:]=a
    np.save('/media/ats/Backup/stivnpys/'+str(mon)+'.npy',aout)
            
#open june through august, save off only OH region and zero out any values greater than 1000 (missing values)
a=np.load('/media/ats/Backup/stivnpys/6.npy')
aa=np.load('/media/ats/Backup/stivnpys/7.npy')
aaa=np.load('/media/ats/Backup/stivnpys/8.npy')
a=np.concatenate((a,aa,aaa),axis=1)
a[a>1000]=0
yps,ypn,xpw,xpe=37.3,42.5,-89,-81.4
yps,ypn,xpw,xpe=38.5,41,-85,-82
sp,pn,wp,ep=65,155,80,170

#it turns out we need to do some further transformation to make this work quickly when only looking at OH
aa=a[:,:,:,sp:pn,wp:ep]
np.save('/media/ats/Backup/stivnpys/jja_oh.npy',a