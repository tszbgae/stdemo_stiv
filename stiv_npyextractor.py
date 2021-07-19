# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pygrib
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

lmon=[31,28,31,30,31,30,31,31,30,31,30,31]

st4ll=Dataset('/media/ats/Backup/stiv/ST4.2005050102.01h.nc')
lats=st4ll.variables['latitude'][:]
lons=st4ll.variables['longitude'][:]

a = pygrib.open('/media/ats/Backup/stiv/2017/ST4.2017050100.01h')
grb = a.select(name='Total Precipitation')[0]
bs=grb.values

latss,latsn,lonsw,lonse=400,600,700,950

def parser(inarr):
    return inarr[latss:latsn,lonsw:lonse]


for yr in range(2002,2018):
    if os.
    for mon in range(5,9):
        for d in range(1,lmon[mon-1]+1):
            

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