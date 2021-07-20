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
from sklearn.cluster import KMeans

lmon=[31,28,31,30,31,30,31,31,30,31,30,31]

st4ll=Dataset('/media/ats/Backup/stiv/ST4.2005050102.01h.nc')
lats=st4ll.variables['latitude'][:]
lons=st4ll.variables['longitude'][:]
latss,latsn,lonsw,lonse=400,600,700,950
def parser(inarr): #function to parse the area give the x-y coords in the line above
    return inarr[latss:latsn,lonsw:lonse]
lon=parser(lons)
lat=parser(lats)


#xlim/ylim bounds for mapping
yps,ypn,xpw,xpe=38.5,41,-85,-82
#bounds for OH region for lat/lon
sp,pn,wp,ep=65,155,80,170
lonaa=lon[sp:pn,wp:ep]
lataa=lat[sp:pn,wp:ep]

latsr=np.linspace(45,35,5)
lonsr=np.linspace(-95,-75,9)

lonsraa,latsraa=np.meshgrid(lonsr,latsr)

aa=np.load('/media/ats/Backup/stivnpys/jja_oh.npy')
ptsx=[-84.14,-84.20,-84.02,-83.81,-84.20,-83.8,-83]
ptsy=[39.63,39.76,39.82,39.92,40.04,40.35,39.96]

#open heights file
hgts=np.load('/media/ats/Backup/data/allhgts02_16.npy')

lvl=0
nclusters=8
#perform clustering of hgts at 500hPa
hgtsr=np.reshape(hgts[:,:,lvl,:,:],(15*91,45))
kmeans = KMeans(n_clusters=nclusters,random_state=0).fit(hgtsr)
labs=kmeans.labels_
cc=np.reshape(kmeans.cluster_centers_,(nclusters,5,9))
amr=np.reshape(np.mean(aa[:,:,17:,:,:],axis=(2)),(1365,90,90))

for x in range(nclusters):
    
    c=amr[labs==x]
    print(len(c))
    cm=np.mean(c,axis=0)
    cmap = plt.get_cmap('jet')
    fig=plt.figure(figsize=(7,10),dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    ax.set_xlim(xpw,xpe) 
    ax.set_ylim(yps,ypn)
    # a=np.where((hours==x) & (conc[:,2]==0))[0]
    #am=np.mean(aa[:,:,17:,:,:],axis=(0,1,2))
    #print(am.max())
    ax.pcolormesh(lonaa,lataa,cm,vmin=.1,vmax=.4,cmap=cmap)
    ax.scatter(ptsx,ptsy,c='r')
    #print(outa[x][0,1],outa[x][:,5])
    # plt.title(str(t) + ' to '+str(t+1))
    # plt.savefig(outdir+str(t)+'_' + str(.75)+'_gt1000.png')
    plt.show()
    plt.close()
    
    fig=plt.figure(figsize=(7,10),dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.STATES)
    
    mn=np.around(cc.min(),decimals=-1)
    mx=np.around(cc.max(),decimals=-1)
    rng=int((mx-mn)/10)
    ax.contourf(lonsraa,latsraa,cc[x],levels=np.linspace(mn,mx,rng),cmap=cmap)
    plt.show()
#plot means of each hour for all days
# for x in range(24):
#     fig=plt.figure(figsize=(7,10),dpi=200)
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.coastlines()
#     ax.add_feature(cfeature.STATES)
#     ax.set_xlim(xpw,xpe) 
#     ax.set_ylim(yps,ypn)
#     # a=np.where((hours==x) & (conc[:,2]==0))[0]
#     am=np.mean(a[:,:,x,:,:],axis=(0,1))
#     print(am.max())
#     ax.pcolormesh(lon,lat,am,vmin=0,vmax=1)
#     #print(outa[x][0,1],outa[x][:,5])
#     # plt.title(str(t) + ' to '+str(t+1))
#     # plt.savefig(outdir+str(t)+'_' + str(.75)+'_gt1000.png')
#     plt.show()
#     plt.close()

#plot mean of all days over OH domain
cmap = plt.get_cmap('jet')
fig=plt.figure(figsize=(7,10),dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.STATES)
ax.set_xlim(xpw,xpe) 
ax.set_ylim(yps,ypn)
# a=np.where((hours==x) & (conc[:,2]==0))[0]
am=np.mean(aa[:,:,17:,:,:],axis=(0,1,2))
print(am.max())
ax.pcolormesh(lonaa,lataa,am,vmin=.1,vmax=.4,cmap=cmap)
ax.scatter(ptsx,ptsy,c='r')
#print(outa[x][0,1],outa[x][:,5])
# plt.title(str(t) + ' to '+str(t+1))
# plt.savefig(outdir+str(t)+'_' + str(.75)+'_gt1000.png')
plt.show()
plt.close()
