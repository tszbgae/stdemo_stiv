# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 11:45:43 2021

@author: bob
"""
import streamlit as st
from sklearn import datasets
import sklearn
from netCDF4 import Dataset
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC as SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%%
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
hgtsr=np.reshape(hgts[:,:,:,:,:],(15*91,4,45))

st.title("streamlit demo")

st.write("""
         # Test Header
         
         Test line
         
         """)

level = st.sidebar.selectbox('select level', ('850 hPa', '500 hPa'))

classifier_name = st.sidebar.selectbox('select classifier', ('Kmeans','DBscan'))

def get_params(classifier_name):
    params=dict()
    if classifier_name == 'Kmeans':
        K=st.sidebar.slider('# clusters',2,9)
        params['n_clusters'] = K
    # elif classifier_name == 'SVM':
    #     c=st.sidebar.slider('C',1,10)
    #     params['C'] = c
    # elif classifier_name == 'Random Forest':
    #     m=st.sidebar.slider('max_depth',2,15)
    #     params['max_depth'] = m
    #     ne=st.sidebar.slider('n_estimators',10,200)
    #     params['n_estimators'] = ne
    return params
@st.cache(suppress_st_warning=True)
def get_class(classifier_name, params):
    if classifier_name == 'Kmeans':
        st.write('using ', classifier_name, 'with ', params['n_clusters'], ' clusters')
        clf=KMeans(n_clusters=params['n_clusters'])
        kmeans=clf.fit(X)
        labs=kmeans.labels_
        cc=np.reshape(kmeans.cluster_centers_,(p['n_clusters'],5,9))
    # elif classifier_name == 'SVM':
    #     clf=SVC(C=params['C'])
    # elif classifier_name == 'Random Forest':
    #     clf=RF(n_estimators=params['n_estimators'], max_depth=params['max_depth'])

    return labs,cc

def get_dataset(level):
    if level == '850 hPa':
        data=hgtsr[:,0,:]
    elif level == '500 hPa':
        data=hgtsr[:,2,:]

    X = data
    st.write(X.shape)

    return X
    
X = get_dataset(level)
p=get_params(classifier_name)
labs,cc=get_class(classifier_name,p)
#xtr,xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=125)
# kmeans=clf.fit(X)
# labs=kmeans.labels_
# cc=np.reshape(kmeans.cluster_centers_,(p['n_clusters'],5,9))
amr=np.reshape(np.mean(aa[:,:,17:,:,:],axis=(2)),(1365,90,90))
#ypr=clf.predict(xte)

st.write('Number of Members in Each Cluster')
clstmbr=[]
for x in range(p['n_clusters']):
    c=amr[labs==x]
    st.write('Cluster '+str(x+1)+': '+str(len(c)))
    clstmbr.append(x+1)


cluster_number = st.sidebar.selectbox('select cluster member', tuple(clstmbr))

st.write('Cluster '+str(cluster_number))


st.set_option('deprecation.showPyplotGlobalUse', False)
cmap = plt.get_cmap('jet')
c=amr[labs==(cluster_number-1)]
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
plt.show()
st.pyplot(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)
cmap = plt.get_cmap('jet')
fig=plt.figure(figsize=(7,10),dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.STATES)
mn=np.around(cc.min(),decimals=-1)
mx=np.around(cc.max(),decimals=-1)
rng=int((mx-mn)/10)
ax.contourf(lonsraa,latsraa,cc[int(cluster_number)-1],levels=np.linspace(mn,mx,rng),cmap=cmap)
plt.show()
st.pyplot(fig)

# my_bar = st.progress(0)
# for percent_complete in range(100):
#     print('al;skdjfsdkljf', percent_complete)
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)
    
# with st.spinner(text='In progress'):
#     for x in range(3):
#         st.spinner(text='in progress '+str(x))
#         time.sleep(5)
# st.success('Done')



    
    