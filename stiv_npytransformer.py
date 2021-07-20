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
            

# lon=parser(lons)
# lat=parser(lats)
# b=parser(bs)

# yps,ypn,xpw,xpe=37.3,42.5,-89,-81.4

# fig=plt.figure(figsize=(7,10),dpi=200)
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# ax.add_feature(cfeature.STATES)
# ax.set_xlim(xpw,xpe) 
# ax.set_ylim(yps,ypn)
# # a=np.where((hours==x) & (conc[:,2]==0))[0]
# ax.pcolormesh(lon,lat,b)
# #print(outa[x][0,1],outa[x][:,5])
# # plt.title(str(t) + ' to '+str(t+1))
# # plt.savefig(outdir+str(t)+'_' + str(.75)+'_gt1000.png')
# plt.show()
# plt.close()