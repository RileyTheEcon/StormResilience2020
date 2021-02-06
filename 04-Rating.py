# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:10:16 2020
@author: RC

Input files:        
Preceeding Script:  
Output files:       
Following Script:   

MUST BE RUN WITH GEOPANDAS ENVIRONMENT
This takes the sample data and produces the ratings for every county.
Note the series of binaries in the variables+libraries section that creates 
the rating system.

"""


# =========================================================================== #
from pandas import *
from numpy import *
from math import *
from geopandas import *
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.ops import cascaded_union,nearest_points
from shapely.geometry import Point,LineString,shape
from matplotlib.colors import LinearSegmentedColormap


# 5A : bNuance==1 & (bForceGroup==1)&(bSigmaGroup==0)
# 5B : bNuance==1 & (bForceGroup==0)&(bSigmaGroup==1)
bNuance     = 1
bMean       = 1                 # Take sum/mean of storm scores
bForceGroup = 0                 # Apply percentiles to unrounded ratings
bSigmaGroup = 1                 # Apply sigma groups
radiusEye   = 30                # Radius of eye, even map no discount      
radiusStorm = 100               # Radius of storm, from center out
radiusDecay = 1.0               # Rate of decay from storm center factor^(radiusDecay)
                                # 1 = linear, >1 = polynomial decay, (0,1) = radical decay
bTexasElevation = 1             # 0 = full Texas, 1 = limited Texas
iTexasElevation = 900.5         # Cut-off for Texas counties by elevation
                                
posInf = float('inf')
negInf = -float('inf')

#   COMPONENTS OF FINAL RATING -- MUST ALSO APPEAR IN LISTTUPPERC
#   Must be six components (some can have 0 weight) if bNuance==0
dictWeights = {#'lnProtectGap'   :1,      
               #'lnTakeUp'       :1,
               #'recovery'       :1,
               #'lnRecovery'     :1,
               #'numInSeason'    :1,
               #'scoreStorm'     :1,
               #'medianIncome'   :1,
               #'ratePoverty'    :1,
               #'rateOwnerOcc'   :1,
               'scoreEcon'      :1,
               'scoreRecovery'  :1,
               'scoreInsurance' :1
               }




#listSigmas     = [-1,0,1,2]
listSigmas     = [-1.036,-0.385,0.385,1.036]
#listSigmas     = [-0.518, -0.1925, 0.1925, 0.518]
listPercentile = [.15,.35,.65,.85]
# listPercentile = [.2,.4,.6,.8]

# [(var name, neg)] 1 := hi values get 1s (hi=bad), 0 := hi values get 5s (hi=good)
#   VARIABLES TO BE RATED AND MAPPED
listTupPerc = [('lnProtectGap',     1),
               ('takeupTotal',      0),
               ('lnTakeUp',         0),
               ('recovery',         1),
               ('lnRecovery',       1),
               #('numInSeason',     1),
               ('scoreStorm',       1),
               ('medianIncome',     0),
               ('ratePoverty',      1),
               #('elevation',       1),
               ('scoreEcon',        0),
               ('scoreRecovery',    0),
               ('scoreInsurance',   0)
               #('rateOwnerOcc',    0)
               ]





#                   [(#,   (lower,  upper])]
#### Insurance
## Insurance Protection Gap (protectGap - 2018)
listProtectGap =    [(1,   23.28,  posInf),
                     (2,   21.60,  23.28),
                     (3,   19.93,  21.6),
                     (4,   18.31,  19.93),
                     (5,   negInf, 18.31)
                     ]

## NFIP Take-Up Rates (takeupTotal - 2018)
listTakeUpRate =    [(1,    negInf, 0),
                     (2,    0,      e**(-6)),
                     (3,    e**(-6),e**(-4)),
                     (4,    e**(-4),e**(-2)),
                     (5,    e**(-2),posInf)
                     ]

#### Storms & Recovery
## Recovery Speed (recovery - 2010-2018)
listRecovery =      [(1,    12,     posInf),
                     (2,    5,      12),
                     (3,    3,      5),
                     (4,    1.9,    3),
                     (5,    negInf, 1.9)
                     ]



# ## Storms Per Season (numInSeason - 2010-2018)
# listStormSeas =     [(1,    .4,     posInf),   #5 storms
#                       (2,    .3,     .4),       #4
#                       (3,    .2,     .3),       #3
#                       (4,    .1,     .2),       #2
#                       (5,    negInf, .1)        #1 or less
#                       ]

# ## Storms Per Season (numInSeason - 2010-2018)
# listStormSeas =     [(1,    .3,     posInf),   #4 or more storms
#                      (2,    .2,     .3),       #3
#                      (3,    .1,     .2),       #2
#                      (4,    0,     .1),       #1
#                      (5,    negInf, 0)        #0
#                      ]


## Storms Per Season (numInSeason - 2010-2018)
listStormSeas =     [(1,    .5,     posInf),   #6 storms
                      (2,    .4,     .5),       #5
                      (3,    .3,     .4),       #4
                      (4,    .2,     .3),       #3
                      (5,    negInf, .2)        #2 or less
                      ]

# ## Storms Per Season (numInSeason - 2010-2018)
# listStormSeas =     [(1,    .1,     posInf),   #6 storms
#                      #(2,    .4,     .5),       #5
#                      (3,    0,     .1),       #4
#                      #(4,    .2,     .3),       #3
#                      (5,    negInf, 0)        #2 or less
#                      ]




#### Socio-Economic
## Median Income (medianIncome - 2018)
listMedianInc =     [(1,    negInf, 25000),
                     (2,    25000,  50000),
                     (3,    50000,  79542),
                     (4,    79542,  130000),
                     (5,    130000, posInf)]

## Poverty Rate (ratePoverty - 2018)
listPovertyRate =   [(1,    20.1,   posInf),
                     (2,    16,     20.1),
                     (3,    8.2,    16),
                     (4,    5.3,    8.2),
                     (5,    negInf, 5.3)]


listRatingGroup = [(1,  0,      1.5),
                   (2,  1.5,    2.5),
                   (3,  2.5,    3.5),
                   (4,  3.5,    4.5),
                   (5,  4.5,    5)]




listRatings = [('lnProtectGap',listProtectGap),
               ('takeupTotal',listTakeUpRate),
               ('recovery',listRecovery),
               ('numInSeason',listStormSeas),
               ('medianIncome',listMedianInc),
               ('ratePoverty',listPovertyRate)]



listStates = ['TX','LA','MS','AL','FL',
              'GA','SC','NC','VA','MD',
              'DE','NJ','NY','CT','RI',
              'MA','NH','ME','PA']



replaceHurr1 = [('abrCounty','LaPorte','La Porte'),
                ('abrCounty','DeKalb','De Kalb'),
                ('abrCounty','LaGrange','La Grange'),
                ('abrCounty','DeSoto','De Soto'),
                ('abrCounty','LaSalle','La Salle'),
                ('abrCounty','LaGrange','La Grange'),
                ('abrCounty','DeWitt','De Witt'),
                ('abrCounty','DuPage','Du Page'),
                ('abrCounty','Doña Ana','Dona Ana'),
                ('abrCounty','LaMoure','La Moure')]

replaceHurr2 = [('stateCode',11,'abrCounty','District of Columbia','Washington'),
                ('stateCode',51,'abrCounty','Newport News','Newport News City'),
                ('stateCode',51,'abrCounty','Suffolk','Suffolk City'),
                ('stateCode',51,'abrCounty','Norfolk','Norfolk City'),
                ('stateCode',51,'abrCounty','Portsmouth','Portsmouth City'),
                ('stateCode',51,'abrCounty','Hampton','Hampton City'),
                ('stateCode',51,'abrCounty','Norton','Norton City'),
                ('stateCode',51,'abrCounty','Virginia Beach','Virginia Beach City'),
                ('stateCode',51,'abrCounty','Fredericksburg','Fredericksburg City'),
                ('stateCode',51,'abrCounty','Manassas Park','Manassas Park City'),
                ('stateCode',51,'abrCounty','Charlottesville','Charlottesville City'),
                ('stateCode',51,'abrCounty','Colonial Heights','Colonial Heights City'),
                ('stateCode',51,'abrCounty','Falls Church','Falls Church City'),
                ('stateCode',51,'abrCounty','Staunton','Staunton City'),
                ('stateCode',51,'abrCounty','Williamsburg','Williamsburg City'),
                ('stateCode',51,'abrCounty','Alexandria','Alexandria City'),
                ('stateCode',51,'abrCounty','Covington','Covington City'),
                ('stateCode',51,'abrCounty','Lynchburg','Lynchburg City'),
                ('stateCode',51,'abrCounty','Manassas','Manassas City'),
                ('stateCode',51,'abrCounty','Chesapeake','Chesapeake City'),
                ('stateCode',51,'abrCounty','Bristol','Bristol City'),
                ('stateCode',51,'abrCounty','Radford','Radford City'),
                ('stateCode',51,'abrCounty','Winchester','Winchester City'),
                ('stateCode',51,'abrCounty','Danville','Danville City'),
                ('stateCode',51,'abrCounty','Waynesboro','Waynesboro City'),
                ('stateCode',51,'abrCounty','Hopewell','Hopewell City'),
                ('stateCode',51,'abrCounty','Buena Vista','Buena Vista City'),
                ('stateCode',51,'abrCounty','Salem','Salem City'),
                ('stateCode',51,'abrCounty','Poquoson','Poquoson City'),
                ('stateCode',51,'abrCounty','Harrisonburg','Harrisonburg City'),
                ('stateCode',51,'abrCounty','Galax','Galax City'),
                ('stateCode',51,'abrCounty','Petersburg','Petersburg City'),
                ('stateCode',51,'abrCounty','Martinsville','Martinsville City'),
                ('stateCode',51,'abrCounty','Emporia','Emporia City'),
                ('stateCode',51,'abrCounty','Lexington','Lexington City')
               ]

# =========================================================================== #










# =========================================================================== #
def merge_track (left,right,by,how='outer',override=0) :
    #   Fixed version!
    left['fromSource'] = 1
    right['fromMerger'] = 1
    
    # assert (how=='left')&(len(right[right[by].duplicated()])>0)&(override==0) ,'Merger not unique'
    dfMerge = left.merge(right,how=how,on=by)
    
    intLeft         = len(left)
    intRight        = len(right)
    intMatch        = len(dfMerge[(dfMerge['fromSource']==1)&
                                  (dfMerge['fromMerger']==1)])
    intLeftError    = len(dfMerge[(dfMerge['fromSource']==1)&
                                  (dfMerge['fromMerger'].isnull())])
    intRightError   = len(dfMerge[(dfMerge['fromSource'].isnull())&
                                  (dfMerge['fromMerger']==1)])
    
    print('In Starting Set: '+str(intLeft)+'\n'
          +'In Mergeing Set: '+str(intRight)+'\n'
          +'Successful Merges: '+str(intMatch)+'\n'
          +'From Start w/o Match: '+str(intLeftError)+'\n'
          +'From Merge w/o Match: '+str(intRightError))
    
    dfMerge['MergeSuccess'] = 0
    dfMerge.loc[(dfMerge['fromSource']==1)&(dfMerge['fromMerger']==1)
                ,'MergeSuccess'] = 1
    
    return dfMerge,intMatch,intLeftError,intRightError
####
def to_rating (x,listRate) :
    r = 0
    for rank in listRate :
        if rank[1]<x<=rank[2] : r = rank[0]
    #   end for
    return r
####
def remove_vars (df,listVar) :
    for var in listVar :
        del df[var]
    return df
####
def print_unique(df,listVars) :
    df = df[listVars].drop_duplicates(subset=listVars)
    df = df.sort_values(by=listVars)
    print(df)
####
def rename_entries_1 (df,listReplace) :
    # Var1 , Entry, New Entry
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1]),
               replace[0]] = replace[2]
    ####
    return df
####
def rename_entries_2 (df,listReplace) :
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1])&
               (df[replace[2]]==replace[3]),
               replace[2]] = replace[4]
    ####
    return df
####
def fix_dupe_location (df) :
    def add_city (df,listReplace) :
        for replace in listReplace :
            df.loc[(df['index']==replace[0]),
                   'abrCounty'] = replace[1]+' City'
        #   end for
        return df
    ####
    # pass shapeMap
    df = df.reset_index()
    mapDupe = df[['index','abrCounty','stateCode']][df[['abrCounty','stateCode']].duplicated()]
    listFix = []
    for index,row in mapDupe.iterrows() :
        listFix.append((row['index'],row['abrCounty'],row['stateCode']))
    #   end for
    df = add_city(df,listFix)
    del df['index']
    return df
####
def subset_by_state (df,listStates) :
    listdfs = []
    for state in listStates :
        listdfs.append(df[df['abrState']==state])
    ####
    df = concat(listdfs)
    return df
####
def plot_hist (df,min_bins=5,max_bins=35,xts=[]) :
    rangeNorm = arange(-3.5,3.5,0.001)
    listBins = list(range(min_bins,max_bins+1))
    listErr = []
    
    mean,stdv,num = df.mean(),std(df),len(df)
    
    for n in listBins :
        dfFit = DataFrame()
        heights,intervals = histogram(df,bins=n)    
        listX = []
        for i in range(1,len(intervals)) :
            listX.append((intervals[i-1]+intervals[i])/2)
        #   endfor
        dfFit['x'] = listX
        dfFit['height'] = heights
        dfFit['norm'] = len(df)*norm.pdf((dfFit['x']-mean)/stdv,0,1)
        dfFit['err'] = (dfFit['height']-dfFit['norm'])**2
        listErr.append(dfFit['err'].mean())
    #   end of bins
    bestBin = listBins[int(where(listErr==min(listErr))[0])]
    fig = plt.figure()
    plt.hist(df,bins=bestBin,edgecolor='white')
    plt.plot(rangeNorm*stdv+mean,len(df)*norm.pdf(rangeNorm,0,1))
    fig.text(.1,.03,'bins = '+str(bestBin)+'  n = '+str(num)
                     +'  mean = '+str(round(mean,3)))
    if len(xts)>0 : plt.xticks(xts)
    plt.title('Distribution of '+df.name)
    plt.savefig('plot/Dist-of-'+df.name+'.png')
####   
# def storm_score (dfSample) :
#     dfStorm = dfSample[(dfSample['bCoastal']==1)&(dfSample['bStorm']==1)
#                        &(dfSample['s1_name'].notnull())]
    
#     # Unique id vars : abrState, county
#     df = DataFrame()
#     for index,row in dfStorm.iterrows():
#         dictRow = {'abrState':row['abrState'],'county':row['county']}
#         if row['numInRecovery']==1:
#             dictRow['vmax'] = row['s1_vmax']
#             dictRow['mslp'] = row['s1_mslp']
#             dictRow['time'] = row['s1_time']
#         if row['numInRecovery']==2:
#             dictRow['vmax'] = row['s2_vmax']
#             dictRow['mslp'] = row['s2_mslp']
#             dictRow['time'] = row['s2_time']
#         if row['numInRecovery']==3:
#             dictRow['vmax'] = row['s3_vmax']
#             dictRow['mslp'] = row['s3_mslp']
#             dictRow['time'] = row['s3_time']
#         df = df.append(dictRow,True)
#     #   end for
#     df['lntime'] = log(df['time'])
#     for var in ['vmax','mslp','lntime'] :
#         df['Z'+var] = (df[var]-mean(df[var]))/std(df[var])
#     df['scoreStorm'] = df['Zvmax'] + df['Zmslp'] + df['Zlntime']
#     for var in ['Zvmax','Zmslp','Zlntime'] :
#         plot_hist(df[var],1)
#     df = df.groupby(['abrState','county'])['scoreStorm'].mean().reset_index()
#     plot_hist(df['scoreStorm'],1)
#     return df
####
def storms_to_one (df) :
    # Pass sample with bCoastal, bStorm, s1_name.notnull() mask
    # Need to keep unique ID year, month, abrState, county
    dfMid = DataFrame()
    for index,row in df.iterrows():
        dictRow = {'abrState':row['abrState'],'county':row['county'],
                   'year':row['year'],'month':row['month'],
                   'bCoastal':row['bCoastal']}
        # If single storm month, run basic storm getting code
        if row['numInMonth']==1:
            if row['numInRecovery']==1:
                dictRow['vmax'] = row['s1_vmax']
                dictRow['mslp'] = row['s1_mslp']
                dictRow['time'] = row['s1_time']
                dictRow['CAT']  = row['s1_CAT']
            if row['numInRecovery']==2:
                dictRow['vmax'] = row['s2_vmax']
                dictRow['mslp'] = row['s2_mslp']
                dictRow['time'] = row['s2_time']
                dictRow['CAT']  = row['s2_CAT']
            if row['numInRecovery']==3:
                dictRow['vmax'] = row['s3_vmax']
                dictRow['mslp'] = row['s3_mslp']
                dictRow['time'] = row['s3_time']
                dictRow['CAT']  = row['s3_CAT']
            dfMid = dfMid.append(dictRow,True)
        # If double storm month, run double code
        elif row['numInMonth']==2 :
            # First storm
            dictRow['vmax'] = row['s1_vmax']
            dictRow['mslp'] = row['s1_mslp']
            dictRow['time'] = row['s1_time']
            dictRow['CAT']  = row['s1_CAT']
            dfMid = dfMid.append(dictRow,True)
            
            # Second storm
            dictRow['vmax'] = row['s2_vmax']
            dictRow['mslp'] = row['s2_mslp']
            dictRow['time'] = row['s2_time']
            dictRow['CAT']  = row['s2_CAT']
            dfMid = dfMid.append(dictRow,True)
    #   end for
    # Translate storms to Zs
    dfMid['lntime'] = dfMid['time'].apply(lambda x : log(x) if x!=0 else None)
    for var in ['vmax','mslp','lntime'] :
        dfMean = mean(dfMid[var][dfMid['bCoastal']==1])
        dfStDv = std(dfMid[var][dfMid['bCoastal']==1])
        dfMid['Z'+var] = (dfMid[var]-dfMean)/dfStDv
    #   end for
    
    # Group by ID, sum of numeric Zs to create dfOut
    dfOut = dfMid.groupby(['year','month','abrState','county'])[
        'Zvmax','Zmslp','Zlntime'].sum().reset_index()
    
    # Merge CAT onto dfOut, keep first dupes (of double storm months)
    dfOut = dfOut.merge(dfMid[['year','month','abrState','county','CAT']][~dfMid[['year','month','abrState','county']].duplicated()],
                        how='left',on=['year','month','abrState','county'])
    
    # Need to merge back onto sample w year, month, abrState, county
    # Compute scoreStorm in main()
    return dfOut[['year','month','abrState','county','Zvmax','Zmslp','Zlntime','CAT']]
####
def bracket_to_ratings (x,listBracket,neg) :
    negInf = -float('inf')
    posInf = float('inf')
    r = 0
    
    if neg==0 :
        if      negInf<x<=listBracket[0]            : r=1
        elif    listBracket[0]<x<=listBracket[1]    : r=2
        elif    listBracket[1]<x<=listBracket[2]    : r=3
        elif    listBracket[2]<x<=listBracket[3]    : r=4
        elif    listBracket[3]<x<=posInf            : r=5
    elif neg==1 :
        if      listBracket[3]<x<=posInf            : r=1
        elif    listBracket[2]<x<=listBracket[3]    : r=2
        elif    listBracket[1]<x<=listBracket[2]    : r=3
        elif    listBracket[0]<x<=listBracket[1]    : r=4
        elif    negInf<x<=listBracket[0]            : r=5
    #   end if
        
    return r
####
def percentile_rating (df,var,listPercentile,neg=0) :
    # Pass df w IV and bCoastal
    
    # Create X-valued brackets from percentiles
    listBracket = []
    for x in listPercentile :
        listBracket.append(df[var][df['bCoastal']==1].quantile(x))
    #   end for    
    
    # Convert df[X] and X-brackets to ratings
    df['rating'] = df[var].apply(bracket_to_ratings,args=(listBracket,neg))
    
    print('\n'+var+'\n'+
          'rank\tbracket\tcount')
    if neg==0 :
        print('1\t(-Inf,'                       +str(round(listBracket[0],3))+']\t'
              +str(df['rating'][(df['rating']==1)&(df['bCoastal']==1)].count()))
        print('2\t('+str(round(listBracket[0],3))+','+str(round(listBracket[1],3))+']\t'
              +str(df['rating'][(df['rating']==2)&(df['bCoastal']==1)].count()))
        print('3\t('+str(round(listBracket[1],3))+','+str(round(listBracket[2],3))+']\t'
              +str(df['rating'][(df['rating']==3)&(df['bCoastal']==1)].count()))
        print('4\t('+str(round(listBracket[2],3))+','+str(round(listBracket[3],3))+']\t'
              +str(df['rating'][(df['rating']==4)&(df['bCoastal']==1)].count()))
        print('5\t('+str(round(listBracket[3],3))+',Inf)\t'
              +str(df['rating'][(df['rating']==5)&(df['bCoastal']==1)].count()))
    elif neg==1 :
        print('1\t('+str(round(listBracket[3],3))+', Inf)\t'
              +str(df['rating'][(df['rating']==1)&(df['bCoastal']==1)].count()))
        print('2\t('+str(round(listBracket[2],3))+','+str(round(listBracket[3],3))+']\t'
              +str(df['rating'][(df['rating']==2)&(df['bCoastal']==1)].count()))
        print('3\t('+str(round(listBracket[1],3))+','+str(round(listBracket[2],3))+']\t'
              +str(df['rating'][(df['rating']==3)&(df['bCoastal']==1)].count()))
        print('4\t('+str(round(listBracket[0],3))+','+str(round(listBracket[1],3))+']\t'
              +str(df['rating'][(df['rating']==4)&(df['bCoastal']==1)].count()))
        print('5\t(-Inf,'                       +str(round(listBracket[0],3))+']\t'
              +str(df['rating'][(df['rating']==5)&(df['bCoastal']==1)].count()))
    #   end if
    
    return df['rating']
####
def nearest_point (x,pts) :
    index = dfHasScore['near-Point']==nearest_points(x,pts)[1]
    nearest = dfHasScore['near-index'][index].values[0]
    return nearest
####
def distance (lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    d = 2 * 3958.8 * asin(sqrt(a)) # radius of earth = 3958.8 miles
    return d
####
def get_distance (tpl) :
    lon1 = tpl[0].x
    lat1 = tpl[0].y
    lon2 = tpl[1].x
    lat2 = tpl[1].y
    
    x = distance(lat1,lon1,lat2,lon2)
    
    return x
####
def storm_decay (x,eye,radius,rate) :
    # x = (stormScore,distanceStorm)
    score = None
    if   x[1]<=eye : score = x[0]
    elif (x[1]>eye)&(x[1]<radius)  : 
        score = x[0]*((radius-x[1])/(radius-eye))**(rate)
    
    
    
    # lambda x : x[0]*((radiusStorm-x[1])/radiusStorm) if x[1]<=radiusStorm
    # else None,axis=1)
    
    return score
####
def sigma_groups (x,listSigmas) :
    r = 0
    if      x<listSigmas[0] : r = 1
    elif listSigmas[0]<=x<listSigmas[1] : r = 2
    elif listSigmas[1]<=x<listSigmas[2] : r = 3
    elif listSigmas[2]<=x<listSigmas[3] : r = 4
    elif listSigmas[3]<=x               : r = 5
    
    return r
####
# =========================================================================== #










# =========================================================================== #
if __name__ =='__main__' :
    
    #   Import data
    dfSample = read_csv('data/working/sampleAllCounties.csv')
    assert len(dfSample)==377040
    
    
    
    
    
    
    #   Create stem with mean of 2018 variables
    dfData = dfSample[dfSample['year']==2018].groupby(['year','abrState','county',
                                                       'bEffectedCounty','bCoastal'])[
        'protectGap','takeupTotal','medianIncome','ratePoverty','rateOwnerOcc'
        ].mean().reset_index()
    assert len(dfData)==3142
    
    
    
    
    
    
    #   Take numInSeason by year, and average by year
    df = dfSample.groupby(['abrState','county','year'])['numInSeason'].max().reset_index()
    df = df.groupby(['abrState','county'])['numInSeason'].mean().reset_index()
    assert len(df)==3142
    dfData = merge_track(dfData,df,['abrState','county'],'left')[0]
    dfData = remove_vars(dfData,['fromSource','fromMerger','MergeSuccess'])
    
    
    
    
    
    
    
    
    
    
    dfSample = merge_track(dfSample,
                           storms_to_one(dfSample[(dfSample['bStorm']==1)]),
                           ['year','month','abrState','county'],'left')[0]
    dfSample['scoreStorm'] = dfSample['Zlntime'] + dfSample['Zvmax'] - dfSample['Zmslp']
    dfSample = remove_vars(dfSample,['fromSource','fromMerger','MergeSuccess'])
    if bMean==0:
        dfData = merge_track(dfData,
                             dfSample[dfSample['scoreStorm'].notnull()].groupby(['abrState','county'])['scoreStorm'].sum().reset_index(),
                             ['abrState','county'],'left')[0]
    else :
        dfData = merge_track(dfData,
                             dfSample.groupby(['abrState','county'])['scoreStorm'].mean().reset_index(),
                             ['abrState','county'],'left')[0]
    dfData = remove_vars(dfData,['fromSource','fromMerger','MergeSuccess'])
    
    
    
    
    
    
    
    
    
    
    #   Import shapefile
    shapeMap = read_file('data/source/county maps/cb_2018_us_county_500k.shp')
    shapeMap = shapeMap[['STATEFP','NAME','geometry']]
    
    #   Rename STATEFP to stateCode
    shapeMap.rename(columns={'STATEFP':'stateCode',
                             'NAME':'abrCounty'},inplace=True)
    shapeMap['stateCode'] = to_numeric(shapeMap['stateCode'],errors='coerce')
    
    #   Add city to cities
    print(shapeMap['abrCounty'][shapeMap[['abrCounty','stateCode']].duplicated(keep=False)])
    shapeMap = fix_dupe_location(shapeMap)
    print(shapeMap['abrCounty'][shapeMap[['abrCounty','stateCode']].duplicated(keep=False)])
    
    #   Remove territories
    shapeMap = shapeMap[(shapeMap['stateCode']!=78)&    # Virgin Islands
                        (shapeMap['stateCode']!=72)&    # Puerto Rico
                        (shapeMap['stateCode']!=69)&    # Mariana Islands
                        (shapeMap['stateCode']!=66)&    # Guam
                        (shapeMap['stateCode']!=60)     # American Samoa
                        ] 
    
    #   Rename abrCounty
    shapeMap = rename_entries_1(shapeMap,replaceHurr1)
    shapeMap = rename_entries_2(shapeMap,replaceHurr2)
    
    
    
    
    
    
    
    
    
    
    #   Add linking cols from censusMaster to shapeMap
    # shapeMap has stateCode + abrCounty ; needs state + county
    print(len(shapeMap))
    dfCensus = read_csv('data/censusMaster.csv')
    dfCounty = dfCensus[['stateCode','abrCounty','abrState','county']].drop_duplicates()
    shapeMap = merge_track(shapeMap,dfCounty,['stateCode','abrCounty'],'left')[0]
    shapeMap = remove_vars(shapeMap,['fromMerger','fromSource','MergeSuccess',
                                     'stateCode',
                                     #'abrCounty'
                                     ])
    print(len(shapeMap))
    
    
    
    
    
    
    
    
    
    
    #   Add TX elevation
    dfTexas = read_csv('data/texas-elevation.csv')
    dfTexas['abrState'] = 'TX'
    shapeMap = merge_track(shapeMap,dfTexas[['abrState','abrCounty','elevation']],
                           ['abrState','abrCounty'],'left')[0]
    shapeMap = remove_vars(shapeMap,['fromMerger','fromSource','MergeSuccess','abrCounty'])
    shapeMap['elevation'].replace(',','',regex=True,inplace=True)
    shapeMap['elevation'] = to_numeric(shapeMap['elevation'],errors='coerce')
    
    
    
    
    
    
    
    
    
    
    #   Add storm score matching
    dfData = merge_track(dfData,
                          shapeMap,['abrState','county'],'left')[0]
    dfData = remove_vars(dfData,['fromMerger','fromSource','MergeSuccess'])
    dfData = GeoDataFrame(dfData,geometry='geometry')
    
    print('Full Map, Has scoreStorm = '+str(dfData['scoreStorm'].notnull().sum()))
    print('Full Map, No  scoreStorm = '+str(dfData['scoreStorm'].isna().sum()))
    print('Coastal Map, Has scoreStorm = '+str(dfData['scoreStorm'][dfData['bCoastal']==1].notnull().sum()))
    print('Coastal Map, No  scoreStorm = '+str(dfData['scoreStorm'][dfData['bCoastal']==1].isna().sum()))
    
    
    dfData['center'] = dfData['geometry'].apply(lambda x : x.representative_point().coords[:])
    listPoints = [Point(coords[0]) for coords in dfData['center']]
    dfData = dfData.rename(columns={'geometry':'outline'})
    dfData = GeoDataFrame(dfData,geometry=listPoints)
    dfData = dfData.reset_index()
    
    dfNoScore = dfData[dfData['scoreStorm'].isna()]
    dfNoScore = remove_vars(dfNoScore,['scoreStorm'])
    
    dfHasScore = dfData[dfData['scoreStorm'].notnull()]
    ptsScore = dfHasScore['geometry'].unary_union
    dfHasScore.rename(columns={'index':'near-index','geometry':'near-Point'},
                      inplace=True)
    dfHasScore['distanceStorm'] = 0
    
    dfNoScore['near-index'] = dfNoScore['geometry'].apply(nearest_point,args=(ptsScore,))
    
    dfNoScore = merge_track(dfNoScore,dfHasScore[['near-index','near-Point','scoreStorm']],
                            ['near-index'],'left')[0]
    
    dfNoScore['distanceStorm'] = dfNoScore[['geometry','near-Point']].apply(get_distance, axis=1)
    
    dfHasScore = remove_vars(dfHasScore,['near-index','near-Point'])
    dfNoScore = remove_vars(dfNoScore,['index','geometry','near-index',
                                       'near-Point','fromMerger',
                                       'fromSource','MergeSuccess'])
    
    dfData = concat([dfHasScore,dfNoScore])
    assert len(dfData)==3142
    assert len(dfData[dfData['bCoastal']==1])==1202
    plot_hist(dfData['distanceStorm'][(dfData['distanceStorm']>0)
                                      #&(dfData['distanceStorm']<radiusStorm)
                                      &(dfData['bCoastal']==1)],1)
    
    # dfData['scoreStorm'] = dfData[['scoreStorm','distanceStorm']].apply(
    #     lambda x : x[0]*((radiusStorm-x[1])/radiusStorm) if x[1]<=radiusStorm
    #     else None,axis=1)
    dfData['scoreStorm'] = dfData[['scoreStorm','distanceStorm']].apply(
        storm_decay,axis=1,args=(radiusEye,radiusStorm,radiusDecay))
    # eye,radius,rate
    
    # Plot storm intensity
    dfTest = DataFrame()
    dfTest['x-axis'] = list(arange(0,radiusStorm+1))
    dfTest['const'] = 1
    dfTest['intensity'] = dfTest[['const','x-axis']].apply(
        storm_decay,axis=1,args=(radiusEye,radiusStorm,radiusDecay))
    fig = plt.figure()
    plt.plot(dfTest['x-axis'],dfTest['intensity'])
    plt.xlabel('Distance from Center (Miles)')
    plt.ylabel('Intensity (%)')
    plt.title('Storm discounting factor')
    plt.savefig('plot/storm-intensity-plot.png')
    
    
    
    
    
    
    
    
    
    
    df = DataFrame(dfSample[dfSample['bStorm']==1].groupby(
        ['abrState','county'])['recovery'].mean().reset_index())
    # assert len(df)==3142
    dfData = merge_track(dfData,df,['abrState','county'],'left')[0]
    dfData = remove_vars(dfData,['fromSource','fromMerger','MergeSuccess'])
    
    # dfData.fillna(0,inplace=True)
    for var in ['recovery','medianIncome','ratePoverty'] :
        dfData[var].fillna(0,inplace=True)
    dfData['takeupTotal']  = dfData['takeupTotal'].apply(lambda x : x if x<1 else 1)
    dfData['lnTakeUp']     = dfData['takeupTotal'].apply(lambda x : log(x) if x!=0 else None)
    dfData['lnProtectGap'] = dfData['protectGap'].apply(lambda x : log(x) if x>0
                                                        else log(dfData['protectGap'][dfData['protectGap']>0].min()))
    dfData['lnRecovery']   = dfData['recovery'].apply(lambda x : log(x) if x!=0 else None)    
    
    if bTexasElevation==1 :
        dfData['bCoastal'] = dfData[['bCoastal','abrState','elevation']].apply(
            lambda x : 0 if (x[1]=='TX')&(x[2]>=iTexasElevation) else x[0],axis=1)
    #   end for
    
    
    # medianIncome pos corr w resilience / povertyRate neg corr w resilience
    # scoreEcon pos corr w resilience
    listScoreVars = [('medianIncome',0),    ('ratePoverty',1),  ('lnRecovery',1),
                     ('scoreStorm',1),      ('lnProtectGap',1), ('lnTakeUp',0),
                     ('recovery',1)]
    for var,neg in listScoreVars :
        dfMean = dfData[var][dfData['bCoastal']==1].mean()
        dfStdv = std(dfData[var][dfData['bCoastal']==1])
        if neg==0 : dfData['Z'+var] = (dfData[var] - dfMean)/(dfStdv)
        else : dfData['Z'+var] = -(dfData[var] - dfMean)/(dfStdv)
    #   end for
    # dfData['scoreEcon']      = dfData[['ZmedianIncome','ZratePoverty']].apply(mean,axis=1)
    # dfData['scoreRecovery']  = dfData[['ZlnRecovery','ZscoreStorm']].apply(mean,axis=1)
    # dfData['scoreInsurance'] = dfData[['ZlnTakeUp','ZlnProtectGap']].apply(mean,axis=1)
    
    
    
    
    
    
    
    
    
    
    #   TAKE DATA >> FUNCTION >> UNROUNDED RATINGS & COMPONENT RATINGS
    dfRatings = dfData[['year','abrState','county','bEffectedCounty',
                        'bCoastal','outline']]
    dfRatings['scoreEcon']      = dfData[['ZmedianIncome','ZratePoverty']].apply(mean,axis=1)
    dfRatings['scoreRecovery']  = dfData[['ZlnRecovery','ZscoreStorm']].apply(mean,axis=1)
    dfRatings['scoreInsurance'] = dfData[['ZlnTakeUp','ZlnProtectGap']].apply(mean,axis=1)
    
    keys = list(dictWeights.keys())
    values = list(dictWeights.values())
    
    if bNuance==0 :
        for tpl in listTupPerc :
            dfRatings[tpl[0]] = percentile_rating(dfData[['bCoastal',tpl[0]]],tpl[0],
                                                  listPercentile,
                                                  neg=tpl[1]
                                                  )
        #   end for
        dfRatings['scoreStorm'] = dfRatings['scoreStorm'].apply(lambda x : x if x!=0 else 5)
        
        dfRatings['Rating'] = (( dfRatings[keys[0]] * values[0]
                                +dfRatings[keys[1]] * values[1]
                                +dfRatings[keys[2]] * values[2]
                                +dfRatings[keys[3]] * values[3]
                                +dfRatings[keys[4]] * values[4]
                                +dfRatings[keys[5]] * values[5]
                                ) / ( values[0] + values[1] + values[2]
                                     +values[3] + values[4] + values[5]))
        dfRatings.loc[(dfRatings['lnProtectGap']==0),'lnProtectGap'] = 1
    else :
        for var,neg in [x for x in listTupPerc if x[0] not in ['scoreEcon','scoreRecovery','scoreInsurance']] :
            dfMean = mean(dfData[var][dfData['bCoastal']==1])
            dfStdv = std(dfData[var][dfData['bCoastal']==1])
            
            dfRatings[var] = (dfData[var] - dfMean)/dfStdv
            if neg==1 : dfRatings[var] = (-1)*dfRatings[var]
        #   end for
        
        dfRatings['Rating'] = dfRatings[keys].apply(mean,axis=1)
        
        for var in [x[0] for x in listTupPerc] :
            dfRatings[var] = dfRatings[var].apply(sigma_groups,args=(listSigmas,))
        #   end for
    #   end if
        
    
    
    
    
    
    
    
    
    
    #   GRAPH DISTRIBUTION OF COMPONENT AND UNROUNDED RATING VARIABLES AS HIST
    df = DataFrame()
    listFreq = ['lnProtectGap','protectGap','lnTakeUp','takeupTotal','lnRecovery',
                'recovery','scoreStorm','medianIncome','ratePoverty']
    
    for var in listFreq :
        #   Graph distributions
        df = dfData[var][(dfData[var].notnull())&(dfData['bCoastal']==1)]
        plot_hist(df,1)
    #   end for     
    #   For Rating
    if bNuance==0 :
        df = dfRatings['Rating'][dfRatings['bCoastal']==1]
        plot_hist(df,1,xts=[1,2,3,4,5])
    else :
        df = dfRatings['Rating'][dfRatings['bCoastal']==1]
        df = df - df.mean() + 3
        plot_hist(df,1,xts=[1,2,3,4,5])
    
    
    
    
    
    
    
    
    
    
    #   TAKE UNROUNDED RATINGS >> FUNCTION >> INTEGER RATINGS
    dfFreq = DataFrame()
    if bForceGroup==1 : 
        dfRatings['rating-round'] = percentile_rating(dfRatings[['bCoastal','Rating']],
                                                      'Rating',
                                                      listPercentile,
                                                      neg=0
                                                      )
    elif bSigmaGroup==1 :
        if bNuance==0 :
            dfMean = mean(dfRatings['Rating'][dfRatings['bCoastal']==1])
            dfStdv = std(dfRatings['Rating'][dfRatings['bCoastal']==1])
        elif bNuance==1 :
            dfMean = 0
            dfStdv = 1
        
        # dfRatings['rating-round'] = dfRatings['Rating'].apply(
        #     lambda x : (x - dfMean)/dfStdv)
        # dfRatings['rating-round'] = dfRatings['rating-round'].apply(sigma_groups,args=(listSigmas,))
        dfRatings['rating-round'] = dfRatings['Rating'].apply(sigma_groups,args=(listSigmas,))
        
        # for var in [x[0] for x in listTupPerc]:
        #     dfRatings[var] = dfRatings[var].apply(sigma_groups,args=(listSigmas,))
        
    else :
        dfRatings['rating-round'] = dfRatings['Rating'].apply(round)
    
    
    
    
    
    
    
    
    
    
    if bNuance==0 :
        for var in ['lnProtectGap','takeupTotal','recovery',
                    'scoreStorm','medianIncome','ratePoverty','rating-round'] :
            if len(dfFreq)==0 : dfFreq = DataFrame(dfRatings[var][dfRatings['bCoastal']==1].value_counts().sort_index().reset_index())
            else : 
                dfFreq = merge_track(dfFreq,
                                        DataFrame(dfRatings[var][dfRatings['bCoastal']==1].value_counts().sort_index().reset_index()),
                                        ['index'],'outer')[0]
                dfFreq = remove_vars(dfFreq,['fromSource','fromMerger','MergeSuccess'])
            #   end if
        #   end for
        print(dfRatings['rating-round'][dfRatings['bCoastal']==1].value_counts().sort_index())
        dfFreq.to_csv('data/rating-freq.csv',index=None)
        dfData.to_csv('data/rating-data.csv',index=None)
        dfRatings.to_csv('data/rating.csv',index=None)
    #   end if
    
    dfGraph = GeoDataFrame(dfRatings[['abrState','county','bCoastal',
                                      'Rating','rating-round']
                                      +[x[0] for x in listTupPerc]],
                           geometry=dfRatings['outline'])
    
    
    
    
    
    
    
    
    
    
    ####    Set boundaries for graphing
    listMapX = [-107,-65]
    listMapY = [24,48]
    intMapScale = 100
    fltX = intMapScale*(listMapX[1]-listMapX[0])/(listMapY[1]-listMapY[0])
    fltY = intMapScale*(listMapY[1]-listMapY[0])/(listMapX[1]-listMapX[0])
     
    #dfGraph = subset_by_state(dfGraph,listStates)
    dfStates = DataFrame()
    for state in list(dfGraph['abrState'][dfGraph['abrState'].notnull()].unique()) :
        dfStates = dfStates.append({'abrState':state,
                                    'geometry':cascaded_union(dfGraph['geometry'][dfGraph['abrState']==state])},True)
    #   end for
    dfStates = GeoDataFrame(dfStates,geometry='geometry')
    listColorMap = [(0,'grey'),
                    (1,'red'),
                    (2,'orange'),
                    (3,'yellow'),
                    (4,'lightgreen'),
                    (5,'green')]
    
    
    
    
    
    
    
    
    
    
    ####    Continuous Colors map
    fig,ax = plt.subplots(1,1,figsize=(fltX,fltY),)
    plt.xlim(listMapX)
    plt.ylim(listMapY)
    plt.tick_params(axis='both',which='both',bottom=True,top=True,
                    labelbottom=False,right=True,left=True,labelleft=False)
    #plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right',size='5%',pad=-2.5)
    cax.tick_params(labelsize=200)
    mapBase = dfGraph[dfGraph['bCoastal']==1].plot(column='Rating',cmap='RdYlGn',
                                                   edgecolor='white',
                                                   figsize=(fltX,fltY),
                                                   legend=True,
                                                   ax=ax,cax=cax,
                                                   #vmin=1,vmax=5,              # turn off if bNuance==1
                                                   #legend_kwds={'orientation':'horizontal'}
                                                   )
    dfGraph[dfGraph['bCoastal']==0].plot(column='Rating',cmap='Greys',
                                         edgecolor='white',
                                         figsize=(fltX,fltY),
                                         #legend=True,
                                         ax=ax,cax=cax,
                                         #vmin=1,vmax=5,                        #turn off if bNuance==1
                                         #legend_kwds={'orientation':'horizontal'}
                                         )
    dfStates.boundary.plot(edgecolor='black',ax=ax)
    mapBase.set_facecolor('xkcd:light blue')
    ####
    plt.savefig('plot/ratings-map-continuous.png',
                bbox_inches='tight')

    
    
    
    
    
    
    
    
    
    ####    Discrete Map, Continuous Greys
    fig,ax = plt.subplots(1,1,figsize=(fltX,fltY),)
    plt.xlim(listMapX)
    plt.ylim(listMapY)
    plt.tick_params(axis='both',which='both',bottom=True,top=True,
                    labelbottom=False,right=True,left=True,labelleft=False)
    # plt.axis('off')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right',size='5%',pad=-3.5)
    # cax.tick_params(labelsize=200)
    mapBase = dfGraph[dfGraph['bCoastal']==0].plot(column='Rating',cmap='Greys',
                                                   edgecolor='white',
                                                   figsize=(fltX,fltY),
                                                   #legend=True,
                                                   ax=ax,
                                                   #cax=cax,
                                                   #vmin=1,vmax=5,
                                                   #legend_kwds={'orientation':'horizontal'}
                                                   )    
    # cmap = LinearSegmentedColormap.from_list('mycmap',listColorMap)
    for tpl in listColorMap :
        dfGraph[(dfGraph['bCoastal']==1)&
                (dfGraph['rating-round']==tpl[0])].plot(color=tpl[1],edgecolor='white',
                                                        figsize=(fltX,fltY),
                                                        legend=True,
                                                        ax=ax,
                                                        #cax=cax,vmin=1,vmax=5,
                                                        #legend_kwds={'orientation':'horizontal'}
                                                        )
    dfStates.boundary.plot(edgecolor='black',ax=ax)
    mapBase.set_facecolor('xkcd:light blue')
    ####
    plt.savefig('plot/ratings-map-discrete-greys.png',
                bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    ####    Discrete Map, Flat Greys
    fig,ax = plt.subplots(1,1,figsize=(fltX,fltY),)
    plt.xlim(listMapX)
    plt.ylim(listMapY)
    plt.tick_params(axis='both',which='both',bottom=True,top=True,
                    labelbottom=False,right=True,left=True,labelleft=False)
    # plt.axis('off')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right',size='5%',pad=-3.5)
    # cax.tick_params(labelsize=200)
    mapBase = dfGraph[dfGraph['bCoastal']==0].plot(color='grey',
                                                   edgecolor='white',
                                                   figsize=(fltX,fltY),
                                                   #legend=True,
                                                   ax=ax,
                                                   #cax=cax,
                                                   #vmin=1,vmax=5,
                                                   #legend_kwds={'orientation':'horizontal'}
                                                   )    
    # cmap = LinearSegmentedColormap.from_list('mycmap',listColorMap)
    for tpl in listColorMap :
        dfGraph[(dfGraph['bCoastal']==1)&
                (dfGraph['rating-round']==tpl[0])].plot(color=tpl[1],edgecolor='white',
                                                        figsize=(fltX,fltY),
                                                        legend=True,
                                                        ax=ax,
                                                        #cax=cax,vmin=1,vmax=5,
                                                        #legend_kwds={'orientation':'horizontal'}
                                                        )
    dfStates.boundary.plot(edgecolor='black',ax=ax)
    mapBase.set_facecolor('xkcd:light blue')
    ####
    plt.savefig('plot/ratings-map-discrete-flat.png',
                bbox_inches='tight')
    
    
    
    
    
    
    
    
    
    
    ####    Discrete Map, Continuous Greys -- Var Specific!
    for var in [x[0] for x in listTupPerc] :
        fig,ax = plt.subplots(1,1,figsize=(fltX,fltY),)
        plt.xlim(listMapX)
        plt.ylim(listMapY)
        plt.tick_params(axis='both',which='both',bottom=True,top=True,
                        labelbottom=False,right=True,left=True,labelleft=False)
        # plt.axis('off')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right',size='5%',pad=-3.5)
        # cax.tick_params(labelsize=200)
        mapBase = dfGraph[dfGraph['bCoastal']==0].plot(column=var,cmap='Greys',
                                                       edgecolor='white',
                                                       figsize=(fltX,fltY),
                                                       #legend=True,
                                                       ax=ax,
                                                       #cax=cax,
                                                       vmin=1,vmax=5,
                                                       #legend_kwds={'orientation':'horizontal'}
                                                       )    
        # cmap = LinearSegmentedColormap.from_list('mycmap',listColorMap)
        for tpl in listColorMap :
            dfGraph[(dfGraph['bCoastal']==1)&
                    (dfGraph[var]==tpl[0])].plot(color=tpl[1],edgecolor='white',
                                                            figsize=(fltX,fltY),
                                                            legend=True,
                                                            ax=ax,
                                                            #cax=cax,vmin=1,vmax=5,
                                                            #legend_kwds={'orientation':'horizontal'}
                                                            )
        #   end for
        dfGraph[(dfGraph['bCoastal']==1)&
                    (dfGraph[var].isna())].plot(color='black',edgecolor='white',
                                                            figsize=(fltX,fltY),
                                                            legend=True,
                                                            ax=ax,
                                                            #cax=cax,vmin=1,vmax=5,
                                                            #legend_kwds={'orientation':'horizontal'}
                                                            )
        dfStates.boundary.plot(edgecolor='black',ax=ax)
        mapBase.set_facecolor('xkcd:light blue')
        ####
        plt.savefig('plot/ratings-map-'+var+'.png',
                    bbox_inches='tight')
    #   end for    
    #   end if
    
    
    
    
    
    
    
    
    
    
    print('\nCoastal Counties w Rating==1\n'+
          str(dfRatings[['abrState','county']][(dfRatings['rating-round']==1)&
                                               (dfRatings['bCoastal']==1)]))
    print('\nCoastal Counties w Rating==5\n'+
          str(dfRatings[['abrState','county']][(dfRatings['rating-round']==5)&
                                               (dfRatings['bCoastal']==1)]))
    
#   end main()
# =========================================================================== #


    # cmap = LinearSegmentedColormap.from_list(
    # 'mycmap', [(0, 'grey'), (1, 'blue')])

    # c.plot(column='color', cmap=cmap)
    
    
    
    
    # modelLasso.mse_path_[int(where(modelLasso.alphas_==modelLasso.alpha_)[0])].mean()
    
    # listFreq = ['lnProtectGap','protectGap','lnTakeUp','takeupTotal','lnRecovery',
    #         'recovery','numInSeason','medianIncome','ratePoverty']
    # rangeNorm = arange(-3.5,3.5,0.001)
    # for var in listFreq :
    #     #   Graph distributions
    #     df = dfData[var][(dfData[var].notnull())&(dfData['bCoastal']==1)]
    #     mean = df.mean()
    #     popstd  = std(df)
    #     plt.figure()
    #     plt.hist(df,bins=10)    # 25
    #     plt.plot(rangeNorm*popstd+mean,
    #              len(df)*norm.pdf(rangeNorm,0,1))
    #     plt.title('Distribution of '+var)
    #     plt.savefig('plot/dist/rate-'+var+'.png')
    # #   end for     
             
    
    
    # dfGraph = subset_by_state(dfGraph,listStates)
    # dfStates = DataFrame()
    # for state in listStates :
    #     dfStates = dfStates.append({'abrState':state,
    #                                 'geometry':cascaded_union(dfGraph['geometry'][dfGraph['abrState']==state])},True)
    # #   end for
    # dfStates = GeoDataFrame(dfStates,geometry='geometry')
    
    # dfGraph['RateGroup'] = dfGraph['Rating'].apply(to_rating,args=(listRatingGroup,))
    
    # dictColors = {1:'Reds',
    #               2:'Orange',
    #               3:'yellows',
    #               4:'greens',
    #               5:'dark greens'}
    
    # fig,ax = plt.subplots(1,1,figsize=(fltX,fltY),)
    # plt.xlim(listMapX)
    # plt.ylim(listMapY)
    # plt.tick_params(axis='both',which='both',bottom=True,top=True,
    #                 labelbottom=False,right=True,left=True,labelleft=False)
    # #plt.axis('off')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right',size='5%',pad=-12)
    # cax.tick_params(labelsize=200)
    # mapBase = dfGraph.plot(column='Rating',cmap='Greens',
    #                        figsize=(fltX,fltY),
    #                        legend=True,
    #                        ax=ax,cax=cax,
    #                        #legend_kwds={'orientation':'horizontal'}
    #                        )
    # dfStates.boundary.plot(edgecolor='black')
    # mapBase.set_facecolor('xkcd:light blue')
    # ####
    # plt.savefig('plot/ratings-5colors.png',
    #             bbox_inches='tight')

