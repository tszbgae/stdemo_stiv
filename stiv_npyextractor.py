# -*- coding: utf-8 -*-
"""
Spyder Editor

this script opens the raw stiv grib files and extracts the midwest region
into hourly precip files in the ../stivnpys folder
"""
import pygrib
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np

lmon=[31,28,31,30,31,30,31,31,30,31,30,31]

st4ll=Dataset('/media/ats/Backup/stiv/ST4.2005050102.01h.nc')
lats=st4ll.variables['latitude'][:]
lons=st4ll.variables['longitude'][:]

a = pygrib.open('/media/ats/Backup/stiv/2017/ST4.2017050100.01h')
grb = a.select(name='Total Precipitation')[0]
bs=grb.values

latss,latsn,lonsw,lonse=400,600,700,950
def parser(inarr): #function to parse the area give the x-y coords in the line above
    return inarr[latss:latsn,lonsw:lonse]
def stringer(innum): #function to turn integer X or XX into a '0X' or 'XX' string
    if innum<10:
        outs='0'+str(innum)
    else:
        outs=str(innum)
    return outs

#loop to extract data from grib to npy
for yr in range(2002,2017):
    if os.path.exists('/home/ats/stivnpys/'+str(yr))==False:
        os.mkdir('/home/ats/stivnpys/'+str(yr))
    for mon in range(8,9):
        if os.path.exists('/home/ats/stivnpys/'+str(yr)+'/'+str(mon))==False:
            os.mkdir('/home/ats/stivnpys/'+str(yr)+'/'+str(mon))
        for d in range(1,lmon[mon-1]+1):
            sd=stringer(d)
            for hr in range(0,24):
                try:
                    sh=stringer(hr)
                    a = pygrib.open('/media/ats/Backup/stiv/'+str(yr)+'/ST4.'+str(yr)+'0'+str(mon)+sd+sh+'.01h')
                    grb = a.select(name='Total Precipitation')[0]
                    bs=grb.values
                    bsm=parser(bs)
                    np.save('/home/ats/stivnpys/'+str(yr)+'/'+str(mon)+'/'+str(yr)+'0'+str(mon)+sd+sh+'.npy',bsm.data)
                    
                except:
                    print('FAIL, '+str(yr)+'0'+str(mon)+sd+sh)
            print(str(yr)+'0'+str(mon)+sd)
            

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