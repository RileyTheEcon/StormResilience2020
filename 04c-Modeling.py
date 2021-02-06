# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:03:15 2021
@author: RC

Input files:        data/working/sampleAllCounties.csv
Output files:       

Modeling focusing on trees and forests.
"""









# =========================================================================== #
from pandas import *
from numpy import log,arange,take,where,std,meshgrid,c_
from sklearn.linear_model import LassoCV,LinearRegression,Lasso
from statsmodels.api import *
from sklearn.neighbors import (NearestNeighbors,KNeighborsClassifier,
                               KNeighborsRegressor,RadiusNeighborsRegressor,
                               NearestCentroid)
from sklearn import tree,metrics
from sklearn.tree import export_text
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from statistics import mean
from itertools import product
from scipy.stats import norm
import seaborn as sns

import winsound
from multiprocessing import *



bDistOut   =1
bRunTree   =1
bRunForest =1
bRunLASSO  =0
bTreeReg   =1
bForestReg =1



dictDVDict = {'recovery'        :'number of months to recovery',
              'recovery24'      :'recovery truncated, values >24 encoded as 25',
              'recovery12'      :'recovery truncated, values >12 encoded as 13',
              'groupRecovery'   :'recovery categorized by quintiles'
              }






listDVCat = ['recovery12','groupRecovery']
listDVNum = ['recovery','recovery24']
# 'recovery','lnrecovery','percentScore'

listNotNeeded = ['year','month','county','abrState','prevProd','Production',
                 'numInMonth','numInSeason','recoverCount',
                 's1_name','s1_CAT','s2_name','s2_CAT']
listCategorical = [('CAT','cat_'),
                   ('numInSeason','season_'),('numInMonth','month_'),
                   ('numInRecovery','recovery_'),('numInSample','sample_')]

# listStorm  = ['s1_vmax','s1_mslp','s1_time','s2_vmax','s2_mslp','s2_time']
listStorm = ['scoreStorm','Zlntime','Zvmax','Zmslp']
listComped = ['takeupTotal','protectGap','totalInsurableValue','insuredValue']
listPolicy = ['policyCount','sumBuildingCoverage','sumBuildingDeductib',
              'sumContentsCoverage','sumContentsDeductib']
listCensus = ['houseTotal','houseMedianValue','medianIncome','perCapitaIncome',
              'rateOccupied','rateOwnerOcc','ratePoverty']
listStormGroup1 = ['cat_TD','cat_TS','cat_HU','cat_EX']
listStormGroup2 = []

# listStormGroup1 = ['s1_EX','s1_TD','s1_TS','s1_H5','s1_H4','s1_H3','s1_H2','s1_H1',
#                    's1_SS','s2_LO','s2_TS','s2_EX','s2_H1',]     # omitted s1_LO , s2_TD
# listStormGroup2 = ['s1_TD','s1_TS','s1_HU','s1_EX','s1_LO',
#                    's2_TD','s2_TS','s2_HU','s2_EX','s2_LO']
listNumGroup    = ['season_2.0','season_3.0','month_2.0',
                   'recovery_2.0','recovery_3.0']    #omitted season_1.0 , month_1.0



posInf = float('inf')
negInf = -float('inf')
listRecovery =      [(1,    12,     posInf),
                     (2,    5,      12),
                     (3,    3,      5),
                     (4,    1.9,    3),
                     (5,    negInf, 1.9)
                     ]


listInteract = []
listInteract = [('lnProtect',           'houseMedianValue'),
                ('rateOwnerOcc',        'sumBuildingDeductib'),
                ('medianIncome',        'ratePoverty'),
                ('houseMedianValue',    'sumContentsCoverage'),
                #('lnProtect',           'lnTakeup'),
                ('medianIncome',        'rateOwnerOcc'),
                ('medianIncome',        'lnTakeup'),
                ('rateOwnerOcc',        'ratePoverty'),
                ('lnMedianIncome',      'lnRatePoverty')
                ]

# 'ratePoverty#medianIncome','sumContentsCoverage#houseMedianValue',
                # 'lnTakeup#lnProtect','medianIncome#rateOwnerOcc',
                # 'medianIncome#lnTakeup','ratePoverty#rateOwnerOcc'

# Hurricane Categories and Counts           Counts (of recovery months)
# TD = Tropical Depression                  783
# TS = Tropical Storm                       835
# HU = Hurricane                            1: 101, 2: 12, 3: 15, 4: 8, 5: 11
# EX = Extratropical Cyclone                1026
# SD = Subtropical Depression               0
# SS = Subtropical Storm                    4
# LO = A low that is not TS, EX or SS       767
# WV = Tropical Wave                        0
# DB = Disturbance                          0
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
def remove_vars (df,listVar) :
    for var in listVar :
        del df[var]
    return df
####
def outreg (dfOutreg,model,modelname,strNote) :
    
    # Get additional info
    listAdd = [('n',model.nobs),
               ('adjR2',model.rsquared_adj),
               ('fstat',float(model.fvalue)),
               ('note',strNote)
               ]
    
    # Get dfParams
    dfParams = model.params
    dfParams = dfParams.reset_index()
    dfParams.columns = ['varname',modelname+'_coef']
    
    # Get dfPvalues
    dfPvalues = model.pvalues
    dfPvalues = dfPvalues.reset_index()
    dfPvalues.columns = ['varname',modelname+'_p']
    
    # Combine/Add results to dfOutreg
    # Start new dfOutreg
    if 'varname' not in list(dfOutreg) :
        # Create df
        dfOutreg = dfParams.merge(dfPvalues,how='outer',on='varname')
        
        # Add additional info
        for tpl in listAdd :
            dfOutreg = dfOutreg.append({'varname':tpl[0],modelname+'_coef':tpl[1]},True)
        #   end for
        
    # Add to existing df
    else :
        # Merge params onto df
        dfOutreg = dfOutreg.merge(dfParams,how='outer',on='varname')
        dfOutreg = dfOutreg.merge(dfPvalues,how='outer',on='varname')
        
        # Add additional info
        for tpl in listAdd :
            dfOutreg.loc[dfOutreg['varname']==tpl[0],
                         modelname+'_coef'] = tpl[1]
        #   end for
        # Move bottom-rows to bottom
        dfBottom = dfOutreg[(dfOutreg['varname']=='n')|
                            (dfOutreg['varname']=='adjR2')|
                            (dfOutreg['varname']=='fstat')|
                            (dfOutreg['varname']=='note')]
        dfOutreg = dfOutreg[(dfOutreg['varname']!='n')&
                            (dfOutreg['varname']!='adjR2')&
                            (dfOutreg['varname']!='fstat')&
                            (dfOutreg['varname']!='note')]
        dfOutreg = concat([dfOutreg,dfBottom])
    #   end if
    
    # Print summary
    print('\nResults for '+modelname+' : ')
    print(model.summary())
    return dfOutreg
####
def to_lnTakeup (x) :
    if x==0 : x = 10**(-10)
    elif x>1: x = 1
    
    r = log(x)
    
    return r
####
def to_rating (x,listRate) :
    r = 0
    for rank in listRate :
        if rank[1]<x<=rank[2] : r = rank[0]
    #   end for
    return r
####
def get_rating_dist (y_hat,df,dfDistOut,modelname,listP,
                     dictModelStats={'mse':0,'adjR2':0,'alpha':0}) :
    #   Add modelname row
    dictRow = {}
    dictRow['col0'] = modelname
    # dictRow['col1'] = '('+str(listP[0])+','+str(listP[1])+','+str(listP[2])+')'
    dfDistOut = dfDistOut.append(dictRow,True)
    
    #   Add Model Ranks
    dfDistOut = dfDistOut.append({'col0':'Rank','col1':'1','col2':'2',
                                  'col3':'3','col4':'4','col5':'5'},True)
    dfDistOut = dfDistOut.append({'col0':'count',
                                  'col1':y_hat[y_hat==1].count(),
                                  'col2':y_hat[y_hat==2].count(),
                                  'col3':y_hat[y_hat==3].count(),
                                  'col4':y_hat[y_hat==4].count(),
                                  'col5':y_hat[y_hat==5].count()}
                                 ,True)
    
    #   Add Dists by var
    for var in list(df) :
        dictRow = {}
        dictRow['col0'] = var
        for x in [1,2,3,4,5] :
            mini = round(df[var][y_hat==x].min(),3)
            p25  = round(df[var][y_hat==x].quantile(.25),3)
            mean = round(df[var][y_hat==x].mean(),3)
            stdv = round(df[var][y_hat==x].std(),3)
            medi = round(df[var][y_hat==x].median(),3)
            p75  = round(df[var][y_hat==x].quantile(.75),3)
            maxi = round(df[var][y_hat==x].max(),3)
            dictRow['col'+str(x)] = ('Min: '    +str(mini)+'\n'
                                     +'P25: '   +str(p25)+'\n'
                                     +'Mean: '  +str(mean)+'\n'
                                     +'StDv: '  +str(stdv)+'\n'
                                     +'Medi: '  +str(medi)+'\n'
                                     +'P75: '   +str(p75)+'\n'
                                     +'Max: '   +str(maxi)
                                     )
        #   end for x
        dfDistOut = dfDistOut.append(dictRow,True)
    #   end for var
    
    # dictRow['cdf'+str(percent)] = (norm.ppf(percent)*dictRow['popStdD'])+dfMean
    
    #   Add Model Stats
    dictRow = {}
    dictRow['col0'],dictRow['col1'] = 'mse',dictModelStats['mse']
    dictRow['col2'],dictRow['col3'] = 'adjR2',dictModelStats['adjR2']
    dictRow['col4'],dictRow['col5'] = 'alpha',dictModelStats['alpha']
    dfDistOut = dfDistOut.append(dictRow,True)
    
    
    #   Add empty linebreak
    dfDistOut = dfDistOut.append({},True)

    return dfDistOut
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
def run_knn (dfY,dfX,listIndices,listParam=range(2,26)) :
    dictMSE = {}    
    nFidelity = 1000
    listNames = list(dfX)
    
    # Convert dfX to standard normal
    dfZ = DataFrame()
    for var in listNames :
        dfMean = mean(dfX[var])
        dfStDv = std(dfX[var])
        dfZ[var] = (dfX[var]-dfMean)/dfStDv
    
    #   Iter over param net
    for param in listParam : 
        listError = []
        
        #   Iter over train,test set
        for fold in listIndices :
            
            #   Fit model on train sets
            knnClass = KNeighborsClassifier(n_neighbors=param,
                                            weights='distance',
                                            ).fit(take(dfZ,fold[0],axis=0),
                                                  take(dfY,fold[0],axis=0))
            
            #   Calculate error on test set. Save to list
            y_pred = knnClass.predict(take(dfZ,fold[1],axis=0))
            listError.append(metrics.accuracy_score(take(dfY,fold[1],axis=0)
                                                    ,y_pred))
            
        #   end for fold
        #   Calculate, save MSE for param
        dictMSE[param] = mean(listError)

    #   end for param
    #   Get best param from CV, 
    bestParam = min(dictMSE,key=dictMSE.get)
    bestMSE = dictMSE[bestParam]
    bestKNNClass = KNeighborsClassifier(n_neighbors=bestParam,
                                        weights='distance',
                                        ).fit(dfZ,dfY)
    
    
    # First Column = X-axis
    x_min,x_max = dfZ[listNames[0]].min(),dfZ[listNames[0]].max()
    # x_min,x_max = -4,4
    
    # Second Column = Y-axis
    y_min,y_max = dfZ[listNames[1]].min(),dfZ[listNames[1]].max()
    # y_min,y_max = -4,4
    
    # Make grid
    xx,yy = meshgrid(arange(x_min-2,x_max+2,(x_max-x_min)/nFidelity),
                     arange(y_min-2,y_max+2,(y_max-y_min)/nFidelity)
                     )
    Zs = bestKNNClass.predict(c_[xx.ravel(),yy.ravel()])
    nColors = len(dfY.unique())
    
    # Make plot
    Zs = Zs.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx,yy,Zs,
                 cmap=ListedColormap(sns.color_palette(None,nColors)))
    for i in range(nColors) :
        sns.scatterplot(x=dfZ[listNames[0]][dfY==i+1],
                        y=dfZ[listNames[1]][dfY==i+1],
                        palette=sns.color_palette(None,nColors)[i],
                        edgecolor='black')
    #   endfor
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    fig.text(.1,.03,'KNN = '+str(bestParam))
    plt.title('Distribution of '+listNames[0]+' & '+listNames[1])
    plt.savefig('plot/trees/04c/KNN-'+listNames[0]+'+'+listNames[1]+'.png')
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
def run_forest_multicore (x) :
    # Takes:
        # x[0] = Keep out var
        # x[1] = DF for dv
        # x[2] = DF for IVs
        # x[3] = list of params
        # x[4] = list of train/test masks
    #setseed(420)
    fltAcc = get_best_forest(x[1],x[2],x[3],x[4])[1]
    print('Complete Var: '+x[0]+'\t\t\tAcc: '+str(fltAcc))
    return (x[0],fltAcc)
####
def get_best_forest (dfY,dfX,listParam,listIndices) :
    dictMSE = {}        
    #   Iter over param net
    for param in listParam : 
        listError = []
        
        #   Iter over train,test set
        for fold in listIndices :
            
            #   Fit model on train sets
            forestClass = RandomForestClassifier(max_depth=param[0],min_samples_split=param[1]
                                                 ).fit(take(dfX,fold[0],axis=0),
                                                       take(dfY,fold[0],axis=0))
            
            #   Calculate error on test set. Save to list
            y_pred = forestClass.predict(take(dfX,fold[1],axis=0))
            listError.append(metrics.accuracy_score(take(dfY,fold[1],axis=0),
                                                    y_pred))
            
        #   end for fold
        #   Calculate, save MSE for param
        dictMSE[param] = mean(listError)

    #   end for param
    #   Report best param and MSE
    bestParam = max(dictMSE,key=dictMSE.get)
    bestMSE = dictMSE[max(dictMSE,key=dictMSE.get)]
    return bestParam,bestMSE
####
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    
    #   Import data & Check distribution of DV
    dfSample = read_csv('data/working/sampleAllCounties.csv')
    
    
    
    
    
    
    
    
    
    
    #   Variable transformation
    df = dfSample.groupby(['abrState','county','year'])['numInSeason'].max().reset_index()
    df = df.groupby(['abrState','county'])['numInSeason'].sum().reset_index()
    df.rename(columns={'numInSeason':'numInSample'},inplace=True)
    dfSample = merge_track(dfSample,df,['abrState','county'],'left')[0]
    dfSample = remove_vars(dfSample,['fromSource','fromMerger','MergeSuccess'])
    
    
    
    
    dfSample = merge_track(dfSample,
                           storms_to_one(dfSample[(dfSample['bStorm']==1)]),
                           ['year','month','abrState','county'],'left')[0]
    dfSample['scoreStorm'] = dfSample['Zlntime'] + dfSample['Zvmax'] + dfSample['Zmslp']
    # dfSample['lntime'] = dfSample['time'].apply(lambda x : log(x) if x!=0 else None)
    # dfSample['scoreStorm'] = get_storm_score(dfSample[['vmax','mslp','lntime']])
    dfSample = remove_vars(dfSample,['fromSource','fromMerger','MergeSuccess'])
    
    
    
    
    # Get indicators for CAT
    # [(var name,prefix)]
    for category in listCategorical :    
        print('\n Value Counts for '+str(category[0]))
        print(get_dummies(dfSample[category[0]]).sum())
        intFirst = 1
        for entry in dfSample[category[0]][dfSample[category[0]].notnull()].unique() :
            dfSample[category[1]+str(entry)] = dfSample[category[0]] == entry
            dfSample[category[1]+str(entry)] = dfSample[category[1]+str(entry)].astype(int)
        #   end for
    #   end for
    listHurr = ['H1','H2','H3','H4','H5']
    listTS   = ['TS','SS']
    dfSample['cat_HU'] = dfSample['CAT'].apply(lambda x : 1 if x in listHurr else 0)
    dfSample['cat_TS'] = dfSample['CAT'].apply(lambda x : 1 if x in listTS else 0)
    dfSample['cat_HU'] = dfSample['CAT'].apply(lambda x : 1 if x in listHurr else 0)
    dfSample['cat_TS'] = dfSample['CAT'].apply(lambda x : 1 if x in listTS else 0)
    
    dfSample['sample_4+'] = dfSample[['sample_4.0','sample_5.0']].apply(lambda x : x[0]+x[1],axis=1)
    
    dfSample['lnrecovery'] = dfSample['recovery'].apply(lambda x : log(x) if x!=0 else None)
    
    for var in ['prevProd','Production','s2_vmax','s2_mslp','s2_time'] :
        dfSample[var] = to_numeric(dfSample[var],errors='coerce')
        dfSample[var] = dfSample[var].fillna(0)
    #   end for
    
    dfSample['percentScore'] = dfSample['percentScore'].apply(
        lambda x : dfSample['percentScore'][dfSample['percentScore']>-float('inf')].min() 
        if x==-float('inf') else x)     # Some neg infs from -Prod/0 for prevProd=0
    
    dfSample['lnTakeup'] = dfSample['takeupTotal'].apply(to_lnTakeup)
    
    dfSample['lnProtect'] = dfSample['protectGap'].apply(lambda x : log(x) if x!=0 else 0)
    dfSample['lnProtect'] = dfSample['lnProtect'].fillna(0)
    
    
    
    
    
    #   Make Subset
    # Make subset for Only effected counties + only effected months
    dfSubset = dfSample[(dfSample['bEffectedCounty']==1)&
                        (dfSample['bStorm']==1)&
                        (dfSample['bCoastal']==1)]
    dfSubset = dfSubset[(dfSubset['recovery'].notnull())].reset_index(drop=True)
    
    print('\n DV Value Counts')
    print(dfSubset['recovery'].value_counts().sort_index())
    
    print('\nCount of Recovery in [1,2) Year Recovery Time :')
    print(dfSubset['recovery'][(dfSubset['recovery']>12)&(dfSubset['recovery']<25)].value_counts().sort_index())
    
    print('\nMean Num in Recovery for [1,2) Year Recovery Time :')
    print(dfSubset[(dfSubset['recovery']>12)&(dfSubset['recovery']<25)].groupby(['recovery'])['numInSeason'].mean())
    
    dfSubset['lnMedianIncome'] = dfSubset['medianIncome'].apply(log)
    dfSubset['lnRatePoverty'] = dfSubset['ratePoverty'].apply(log)
    
    
    
    
    
    # 'sumBuildingCoverage','sumBuildingDeductib','sumContentsCoverage','sumContentsDeductib',
    listScoreVars = [('medianIncome',0),    
                     ('ratePoverty',1),  
                     ('lnProtect',1), 
                     ('lnTakeup',0),
                     ('sumContentsCoverage',0),
                     ('sumBuildingCoverage',0),
                     ('sumContentsDeductib',0),
                     ('sumBuildingDeductib',0),
                     ('rateOwnerOcc',0),
                     ('houseMedianValue',0)
                     ]
    for var,neg in listScoreVars :
        dfMean = dfSubset[var][dfSubset['bCoastal']==1].mean()
        dfStdv = std(dfSubset[var][dfSubset['bCoastal']==1])
        if neg==0 : dfSubset['Z'+var] = (dfSubset[var] - dfMean)/(dfStdv)
        else : dfSubset['Z'+var] = -(dfSubset[var] - dfMean)/(dfStdv)
    #   end for
    dfSubset['scoreEcon']      = dfSubset[['ZmedianIncome','ZratePoverty']].apply(mean,axis=1)
    dfSubset['scoreInsurance'] = dfSubset[['ZlnTakeup','ZlnProtect']].apply(mean,axis=1)
    dfSubset['scoreCoverage']  = dfSubset[['ZsumContentsCoverage',
                                           'ZsumBuildingCoverage']].apply(mean,axis=1)
    dfSubset['scoreDeductib']  = dfSubset[['ZsumContentsDeductib',
                                           'ZsumBuildingDeductib']].apply(mean,axis=1)
    dfSubset['scoreHousing']   = dfSubset[['ZrateOwnerOcc',
                                           'ZhouseMedianValue']].apply(mean,axis=1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    randSeed = 210
    listIndices = [(dfTrain,dfTest) for dfTrain,dfTest 
                   in KFold(n_splits=10,shuffle=True).split(dfSubset)]
    
    
    
    
    
    
    
    
    
    
    
    ### Make interactions
    # ratePoverty           x   medianIncome
    # sumContentsCoverage   x   houseMedianValue
    # lnTakeup              x   lnProtect
    # medianIncome          x   rateOwnerOcc
    # medianIncome          x   lnTakeup
    # ratePoverty           x   rateOwnerOcc
    listCombo = []
    dfSubset['groupRecovery'] = percentile_rating(dfSubset[['bCoastal','recovery']],
                                                  'recovery',
                                                  [.2,.4,.6,.8],
                                                  neg=0)
    
    for var1,var2 in listInteract+[
            ('scoreEcon','scoreStorm'),('scoreEcon','scoreInsurance'),
            ('scoreStorm','scoreInsurance'),('lnProtect','lnTakeup')] :
        dfSubset[var1+'#'+var2] = dfSubset[var1] * dfSubset[var2]
        # run_knn(dfSubset['groupRecovery'],dfSubset[[var1,var2]],listIndices)
        listCombo = listCombo + [var1+'#'+var2]
    #   endfor
    
    
    # run_knn(dfSubset['recovery12'],dfSubset[['ratePoverty','medianIncome']],listIndices)
    
    
    # for var1,var2 in listInteract :
    #     run_knn(dfSubset['groupRecovery'],dfSubset[[var1,var2]],listIndices)
    
    
    # dfSubset['lnMedianIncome'] = dfSubset['medianIncome'].apply(log)
    # dfSubset['lnRatePoverty'] = dfSubset['ratePoverty'].apply(log)
    
    
    
    # from random import randint
    # Zs = [randint(1,5) for i in range(100)]
    
    
    
    
    # for var1 in ['houseMedianValue','ratePoverty','scoreStorm'] :
    #     run_knn(dfSubset['groupRecovery'],dfSubset[[var1,'numInSample']],listIndices)
    # #   endfor
    
    
    
    
    
    
    
    
    
    
    
    ####    Start ML Model fitting and classification
    strLog = 'Start of Model log:\n'
    # listDVCat = ['recovery','recovery12','recovery24']
    # listDVNum = ['recovery','lnrecovery','percentScore']
    
    #   Run with takeupTotal if dv!=lnrecovery, else lnTakeup
    #   Run with protectGap if dv!=lnrecovery, else lnProtect
    listVars = ['sample_1.0','sample_2.0','sample_3.0','sample_4+',     # 0.3041407867494824
                
                'scoreCoverage',
                'sumBuildingCoverage',
                'sumContentsCoverage',
                
                'scoreDeductib',
                'sumBuildingDeductib',
                'sumContentsDeductib',
                                         
                'scoreStorm',
                
                'scoreEcon',                            # 0.3139751552795031
                'medianIncome',
                'ratePoverty',                          # 0.304120082815735
                
                'scoreHousing',
                'rateOwnerOcc',
                'houseMedianValue',      # 0.30124223602484473
                
                'scoreInsurance',                       # 0.3200414078674948
                'takeupTotal',
                'protectGap',             # 0.30420289855072463
                
                # 'Zlntime','Zvmax','Zmslp',
                # 'cat_TD','cat_TS','cat_HU','cat_EX',
                # 's1_vmax','s1_mslp','s1_time','s1_TD','s1_TS','s1_HU','s1_EX',
                # 'ratePoverty#medianIncome','sumContentsCoverage#houseMedianValue',
                # 'lnTakeup#lnProtect','medianIncome#rateOwnerOcc',
                # 'medianIncome#lnTakeup','ratePoverty#rateOwnerOcc'
                ]
    listVars = listVars+listCombo
    #s1_LO omitted for multicolinearity # 'month_2.0','season_2.0','season_3.0',
    # takeupTotal
    #   Generate list of train/test indices [(train,test),...]
    
    
    dfDistOut = DataFrame()
    
    
    # ####    OLS Regs
    # dfOutReg = DataFrame()
    # for dv in listDVCat+['lnrecovery','percentScore'] :
    #     if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
    #     else : listIV = ['lnTakeup','lnProtect']+listVars
    #     modelMultiReg = OLS(dfSubset[dv],add_constant(dfSubset[listIV])
    #                          ).fit()
    #     # dfOutreg = outreg(dfOutreg,modelMultiReg,dv,'no clustered std err')
    # #   end for dv
    # # dfOutreg.to_csv('log/multireg-coefs.csv',index=None)
    
    
    
    
    
    ####    LASSO Model
    if bRunLASSO==1 :
        print('Start LASSO code')
        strLog = strLog+'Start LASSO code\n'
        # Start coef df
        dfCoef = DataFrame()
        # dfCoef['vars'] = listIV
        
        for dv in listDVNum :
            if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
            else : listIV = ['lnTakeup','lnProtect']+listVars
            # Get vars
            df = DataFrame()
            df['vars'] = listIV
            
            # Set params
            maxDepth = 10
            lastAlpha = 1
            currentAlpha = -1
            currentIter = 0
            
            # define upper lower
            upper = 1
            lower = 0
            # start while loop
            while ((currentAlpha==0)|(lastAlpha!=currentAlpha))&(currentIter<maxDepth) :
                lastAlpha = currentAlpha
                # fit model
                modelLasso = LassoCV(alphas=arange(lower,upper,((upper-lower)/10)),
                                 cv=RepeatedKFold(n_splits=10,n_repeats=3,random_state=randSeed),
                                 max_iter=10000,
                                 n_jobs=-1).fit(dfSubset[listIV],dfSubset[dv])
                # get best lambda and its index
                currentAlpha = modelLasso.alpha_
                bestIndex = int(where(modelLasso.alphas_==modelLasso.alpha_)[0])
                
                # get new upper lower
                if bestIndex==0 : upper,lower = modelLasso.alphas_[0],modelLasso.alphas_[1]
                elif bestIndex==9 : upper,lower = modelLasso.alphas_[8],modelLasso.alphas_[9]
                else : upper,lower = modelLasso.alphas_[bestIndex-1],modelLasso.alphas_[bestIndex+1]
            
                currentIter+=1
            #   end while
            bestLasso = Lasso(alpha=modelLasso.alpha_).fit(dfSubset[listIV],
                                                           dfSubset[dv])
            
            df[dv] = bestLasso.coef_
            if len(dfCoef)==0 : dfCoef=df
            else : dfCoef = dfCoef.merge(df,'outer','vars')
            
            score = bestLasso.score(dfSubset[listIV],dfSubset[dv])
            adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
                                                          
            if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
            else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
            dfDistOut = get_rating_dist(DataFrame(bestLasso.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                        dfSubset[listDistr],
                                        dfDistOut,
                                        'Lasso : '+dv,
                                        [0.16,0.5,0.84],
                                        {'mse':str(modelLasso.mse_path_[int(where(modelLasso.alphas_==modelLasso.alpha_)[0])].mean()),
                                         'adjR2':str(adjR2),
                                         'alpha':modelLasso.alpha_})
            
            # listDistr = []
            # y_hat = to_rating(bestTreeClass.predict(dfSubset[listIV]),listRecovery)
            # def get_rating_dist (y_hat,df,dfDistOut,modelname,listP,
            #                      dictModelStats={'mse':0,'adjR2':0}) :
    
            
            
            print('For lasso with dv '+dv+' best alpha='+str(modelLasso.alpha_)+' with MSE='
                  +str(modelLasso.mse_path_[int(where(modelLasso.alphas_==modelLasso.alpha_)[0])].mean())
                  +' adjR2='+str(adjR2))
            strLog = strLog+str('For dv '+dv+' best alpha='+str(modelLasso.alpha_)+' with MSE='
                  +str(modelLasso.mse_path_[int(where(modelLasso.alphas_==modelLasso.alpha_)[0])].mean())
                       +' adjR2='+str(adjR2)+'\n')
        #   end for vars
        print(dfCoef)
        dfCoef.to_csv('log/lasso-coefs.csv',index=None)
        strLog=strLog+'Lasso coefs exporting to lasso-coefs.csv\n'
    #   endif
    
    
    
    
        





    
    ####    Trees
    #   Make param list
    listPercent = [2/len(dfSubset),0.01,0.05,0.1]
    listSampleM = [round(x*len(dfSubset)) for x in listPercent]
    listParam = list(product(range(2,6),listSampleM))
    print('\nStart Trees code')
    strLog = strLog+'\nStart Trees code\n'
    dictTextTrees = {}
    strTextTrees = ('random seed = '+str(randSeed)+'\n'+
                    'Max Depths considered : '+
                    str(min([x[0] for x in listParam]))+
                    ' to '+str(max([x[0] for x in listParam]))+' , '+
                    'Min Sample Split considered : '+str(listSampleM)+'\n\n'
                    )
    
    
    





    dictVarModels = {'groupRecovery'    :['sample_1.0','sample_2.0','sample_3.0',
                                          'sample_4+','scoreStorm',
                                          'scoreEcon',
                                          'scoreInsurance',
                                          'sumContentsCoverage','sumBuildingCoverage',
                                          'scoreDeductib',
                                          'scoreHousing',
                                          'rateOwnerOcc#sumBuildingDeductib',
                                          'houseMedianValue#sumContentsCoverage',
                                          'medianIncome#rateOwnerOcc',
                                          'medianIncome#lnTakeup',
                                          'rateOwnerOcc#ratePoverty',
                                          'scoreEcon#scoreStorm',
                                          'scoreEcon#scoreInsurance',
                                          'scoreStorm#scoreInsurance',
                                          'lnProtect#lnTakeup'],
                     
                     'recovery12'       :['sample_1.0','sample_2.0','sample_3.0',
                                          'sample_4+','scoreStorm',
                                          'scoreEcon',
                                          'takeupTotal','protectGap',
                                          'scoreCoverage',
                                          'scoreDeductib',
                                          'rateOwnerOcc','houseMedianValue',
                                          'rateOwnerOcc#sumBuildingDeductib',
                                          'houseMedianValue#sumContentsCoverage',
                                          'medianIncome#rateOwnerOcc',
                                          'medianIncome#lnTakeup',
                                          'rateOwnerOcc#ratePoverty',
                                          'scoreEcon#scoreStorm',
                                          'scoreEcon#scoreInsurance',
                                          'scoreStorm#scoreInsurance',
                                          'lnProtect#lnTakeup'],
                     
                     'recovery24'       :['sample_1.0','sample_2.0','sample_3.0',
                                          'sample_4+','scoreStorm',
                                          'medianIncome','ratePoverty',
                                          'takeupTotal','protectGap',
                                          'scoreCoverage',
                                          'scoreDeductib',
                                          'rateOwnerOcc','houseMedianValue',
                                          'rateOwnerOcc#sumBuildingDeductib',
                                          'houseMedianValue#sumContentsCoverage',
                                          'medianIncome#rateOwnerOcc',
                                          'medianIncome#lnTakeup',
                                          'rateOwnerOcc#ratePoverty',
                                          'scoreEcon#scoreStorm',
                                          'scoreEcon#scoreInsurance',
                                          'scoreStorm#scoreInsurance',
                                          'lnProtect#lnTakeup'],
                     
                     'recovery'         :['sample_1.0','sample_2.0','sample_3.0',
                                          'sample_4+','scoreStorm',
                                          'scoreEcon',
                                          'takeupTotal','protectGap',
                                          'scoreCoverage',
                                          'scoreDeductib',
                                          'rateOwnerOcc','houseMedianValue',
                                          'rateOwnerOcc#sumBuildingDeductib',
                                          'houseMedianValue#sumContentsCoverage',
                                          'medianIncome#rateOwnerOcc',
                                          'medianIncome#lnTakeup',
                                          'rateOwnerOcc#ratePoverty',
                                          'scoreEcon#scoreStorm',
                                          'scoreEcon#scoreInsurance',
                                          'scoreStorm#scoreInsurance',
                                          'lnProtect#lnTakeup']
                     }
    
    
    
    
    
    
    
    
    
    # bRunTree = 1


    if bRunTree==1:
        ####    Classification Trees
        for dv in listDVCat :
            # if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
            # else : listIV = ['lnTakeup','lnProtect']+listVars
            listIV = dictVarModels[dv]
            dictMSE = {}        
            #   Iter over param net
            for param in listParam : 
                listError = []
                
                #   Iter over train,test set
                for fold in listIndices :
                    
                    #   Fit model on train sets
                    treeClass = tree.DecisionTreeClassifier(max_depth=param[0],min_samples_split=param[1],
                                                   ).fit(take(dfSubset[listIV],fold[0],axis=0),
                                                          take(dfSubset[dv],fold[0],axis=0))
                    
                    #   Calculate error on test set. Save to list
                    y_pred = treeClass.predict(take(dfSubset[listIV],fold[1],axis=0))
                    listError.append(metrics.accuracy_score(take(dfSubset[dv],fold[1],axis=0)
                                                            ,y_pred))
                    
                #   end for fold
                #   Calculate, save MSE for param
                dictMSE[param] = mean(listError)
        
            #   end for param
            #   Report best param and MSE
            bestParam = max(dictMSE,key=dictMSE.get)
            bestMSE = dictMSE[max(dictMSE,key=dictMSE.get)]
            bestTreeClass = tree.DecisionTreeClassifier(max_depth=bestParam[0],
                                                        min_samples_split=bestParam[1],
                                                        random_state=randSeed
                                                        ).fit(dfSubset[listIV],dfSubset[dv])
            
            score = bestTreeClass.score(dfSubset[listIV],dfSubset[dv])
            adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
            
            fig, ax = plt.subplots(figsize=(100,50))
            tree.plot_tree(bestTreeClass,fontsize=15,feature_names=listIV,)
            plt.savefig('plot/trees/04c/bestTreeClass-'+str(dv)+'.png')
            
            if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
            else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
            dfDistOut = get_rating_dist(DataFrame(bestTreeClass.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                        dfSubset[listDistr],
                                        dfDistOut,
                                        'TreeClass : '+dv,
                                        [0.16,0.5,0.84],
                                        {'mse':str(bestMSE),
                                         'adjR2':str(adjR2),
                                         'alpha':bestParam})
            
            strTextTrees =  (strTextTrees+dv+' : '+dictDVDict[dv]+'\n'+
                             'Class. Tree: '+
                             'Max depth = '+str(bestParam[0])+
                             ' , Min Sample Split = '+str(bestParam[1])+'\n'+
                             export_text(bestTreeClass,feature_names=listIV)+'\n'
                            )
                            
            dictTextTrees[dv] = export_text(bestTreeClass,feature_names=listIV)
            print('For Classification Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
            strLog = strLog+'For Classification Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
        #   end for dv    
    #   endif
    
    
    
    
    
    if bRunForest==1 :
        ####    Decision Forest
        for dv in listDVCat :
            # if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
            # else : listIV = ['lnTakeup','lnProtect']+listVars
            listIV = dictVarModels[dv]
            dictMSE = {}        
            #   Iter over param net
            for param in listParam : 
                listError = []
                
                #   Iter over train,test set
                for fold in listIndices :
                    
                    #   Fit model on train sets
                    forestClass = RandomForestClassifier(max_depth=param[0],min_samples_split=param[1]
                                                       ).fit(take(dfSubset[listIV],fold[0],axis=0),
                                                             take(dfSubset[dv],fold[0],axis=0))
                    
                    #   Calculate error on test set. Save to list
                    y_pred = forestClass.predict(take(dfSubset[listIV],fold[1],axis=0))
                    listError.append(metrics.accuracy_score(take(dfSubset[dv],fold[1],axis=0)
                                                            ,y_pred))
                    
                #   end for fold
                #   Calculate, save MSE for param
                dictMSE[param] = mean(listError)
        
            #   end for param
            #   Report best param and MSE
            bestParam = max(dictMSE,key=dictMSE.get)
            bestMSE = dictMSE[max(dictMSE,key=dictMSE.get)]
            
            bestForestClass = RandomForestClassifier(max_depth=bestParam[0],min_samples_split=bestParam[1]
                                           ).fit(dfSubset[listIV],dfSubset[dv])
            score = bestForestClass.score(dfSubset[listIV],dfSubset[dv])
            adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
            
            fig, ax = plt.subplots(figsize=(100,50))
            tree.plot_tree(bestForestClass.estimators_[0],fontsize=15,feature_names=listIV,)
            plt.savefig('plot/trees/04c/bestForestClass-'+str(dv)+'.png')
            
            
            if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
            else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
            dfDistOut = get_rating_dist(DataFrame(bestForestClass.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                        dfSubset[listDistr],
                                        dfDistOut,
                                        'ForestClass : '+dv,
                                        [0.16,0.5,0.84],
                                        {'mse':str(bestMSE),
                                         'adjR2':str(adjR2),
                                         'alpha':bestParam})
            
            # txtBestForest = export_text(bestForestClass,feature_names=listIV)
            print('For Classification Forest, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
            strLog = strLog+'For Classification Forest, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
        #   end for dv    
    #   endif
    
    
    
    
    
    
    
    
    
    
    if bTreeReg==1 :
        for dv in listDVNum :
            # if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
            # else : listIV = ['lnTakeup','lnProtect']+listVars
            listIV = dictVarModels[dv]
            dictMSE = {}        
            #   Iter over param net
            for param in listParam : 
                listError = []
                
                #   Iter over train,test set
                for fold in listIndices :
                    
                    #   Fit model on train sets
                    treeReg = tree.DecisionTreeRegressor(max_depth=param[0],min_samples_split=param[1],
                                                         ).fit(take(dfSubset[listIV],fold[0],axis=0),
                                                               take(dfSubset[dv],fold[0],axis=0))
                    
                    #   Calculate error on test set. Save to list
                    y_pred = treeReg.predict(take(dfSubset[listIV],fold[1],axis=0))
                    listError.append(metrics.mean_squared_error(take(dfSubset[dv],fold[1],axis=0)
                                                                ,y_pred))
                    
                #   end for fold
                #   Calculate, save MSE for param
                dictMSE[param] = mean(listError)
        
            #   end for param
            #   Report best param and MSE
            bestParam = min(dictMSE,key=dictMSE.get)
            bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
            
            bestTreeReg = tree.DecisionTreeRegressor(max_depth=bestParam[0],min_samples_split=bestParam[1]
                                                     ).fit(dfSubset[listIV],dfSubset[dv])
            score = bestTreeReg.score(dfSubset[listIV],dfSubset[dv])
            adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
            
            fig, ax = plt.subplots(figsize=(100,50))
            tree.plot_tree(bestTreeReg,fontsize=15,feature_names=listIV,)
            plt.savefig('plot/trees/04c/bestRegTree-'+str(dv)+'.png')
            
            if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
            else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
            dfDistOut = get_rating_dist(DataFrame(bestTreeReg.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                        dfSubset[listDistr],
                                        dfDistOut,
                                        'TreeReg : '+dv,
                                        [0.16,0.5,0.84],
                                        {'mse':str(bestMSE),
                                         'adjR2':str(adjR2),
                                         'alpha':bestParam})
            
            strTextTrees =  (strTextTrees+dv+' : '+dictDVDict[dv]+'\n'+
                             'Reg. Tree: '+
                             'Max depth = '+str(bestParam[0])+
                             ' , Min Sample Split = '+str(bestParam[1])+'\n'+
                             export_text(bestTreeReg,feature_names=listIV)+'\n'
                             )
            
            dictTextTrees[dv] = export_text(bestTreeReg,feature_names=listIV)
            print('For Regression Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
            strLog = strLog+'For Regression Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
        #   end for dv  
    #   endif
    
    
    
    
    
    
    
    
    
    
    if bForestReg==1 :
        for dv in listDVNum :
            # if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
            # else : listIV = ['lnTakeup','lnProtect']+listVars
            listIV = dictVarModels[dv]
            dictMSE = {}        
            #   Iter over param net
            for param in listParam : 
                listError = []
                
                #   Iter over train,test set
                for fold in listIndices :
                    
                    #   Fit model on train sets
                    forestReg = RandomForestRegressor(max_depth=param[0],min_samples_split=param[1]
                                                       ).fit(take(dfSubset[listIV],fold[0],axis=0),
                                                             take(dfSubset[dv],fold[0],axis=0))
                    
                    #   Calculate error on test set. Save to list
                    y_pred = forestReg.predict(take(dfSubset[listIV],fold[1],axis=0))
                    listError.append(metrics.mean_squared_error(take(dfSubset[dv],fold[1],axis=0)
                                                            ,y_pred))
                    
                #   end for fold
                #   Calculate, save MSE for param
                dictMSE[param] = mean(listError)
        
            #   end for param
            #   Report best param and MSE
            bestParam = min(dictMSE,key=dictMSE.get)
            bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
            
            bestForestReg = RandomForestRegressor(max_depth=bestParam[0],min_samples_split=bestParam[1]
                                           ).fit(dfSubset[listIV],dfSubset[dv])
            score = bestForestReg.score(dfSubset[listIV],dfSubset[dv])
            adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
            
            fig, ax = plt.subplots(figsize=(100,50))
            tree.plot_tree(bestForestReg.estimators_[0],fontsize=15,feature_names=listIV,)
            plt.savefig('plot/trees/04c/bestRegForest-'+str(dv)+'.png')
            
            if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
            else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
            dfDistOut = get_rating_dist(DataFrame(bestForestReg.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                        dfSubset[listDistr],
                                        dfDistOut,
                                        'ForestReg : '+dv,
                                        [0.16,0.5,0.84],
                                        {'mse':str(bestMSE),
                                         'adjR2':str(adjR2),
                                         'alpha':bestParam})
            
            print('For Regression Forest, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
            strLog = strLog+'For Regression Forest, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
        #   end for dv    
    #   endif
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ####    Output logs
    if bDistOut==1 :
        print(strLog)
        dfDistOut.to_csv('log/04c-distribution-out.csv',index=None,header=False)
    #   endif
    print(strTextTrees)
    fileOut = open('log/text trees.txt','w')
    fileOut.write(strTextTrees)
    fileOut.close()
    
    
    
    
    
    
    
    def get_best_tree (dfY,dfX,listParam,listIndices) :
        dictMSE = {}        
        #   Iter over param net
        for param in listParam : 
            listError = []
            
            #   Iter over train,test set
            for fold in listIndices :
                
                #   Fit model on train sets
                treeClass = tree.DecisionTreeClassifier(max_depth=param[0],min_samples_split=param[1]
                                                        ).fit(take(dfX,fold[0],axis=0),
                                                              take(dfY,fold[0],axis=0))
                
                #   Calculate error on test set. Save to list
                y_pred = treeClass.predict(take(dfX,fold[1],axis=0))
                listError.append(metrics.accuracy_score(take(dfY,fold[1],axis=0),
                                                        y_pred))
                
            #   end for fold
            #   Calculate, save MSE for param
            dictMSE[param] = mean(listError)
    
        #   end for param
        #   Report best param and MSE
        bestParam = max(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[max(dictMSE,key=dictMSE.get)]
        return bestParam,bestMSE
    ####
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    #   Step-Wise Removing Vars
    # Take in:
    # listIV    , listParam
    # listDVCat , listIndices
    # listDVCat = ['recovery', 'recovery12', 'recovery24', 'groupRecovery']
    
    bRunSwap = 0
    dv = 'groupRecovery'
    
    # 'groupRecovery'   = scoreEcon + scoreInsurance + 
    # 'recovery12'      = scoreEcon + scoreInsurance
    # 'recovery24'      = scoreInsurance
    # 'recovery'        = scoreEcon + scoreCoverage + scoreHousing
    
    #   Check collinear variables
    listBaseVars = ['sample_1.0','sample_2.0','sample_3.0','sample_4+','scoreStorm',
                    'medianIncome','ratePoverty','takeupTotal','protectGap',
                    'sumContentsCoverage','sumBuildingCoverage',
                    'sumContentsDeductib','sumBuildingDeductib',
                    'rateOwnerOcc','houseMedianValue'
                    ]+listCombo   # base list of vars
    
    listSwaps = [(['medianIncome','ratePoverty'],['scoreEcon']),                        # 0
                 (['takeupTotal','protectGap'],['scoreInsurance']),                     # 1
                 (['sumContentsCoverage','sumBuildingCoverage'],['scoreCoverage']),     # 2
                 (['sumContentsDeductib','sumBuildingDeductib'],['scoreDeductib']),     # 3
                 (['rateOwnerOcc','houseMedianValue'],['scoreHousing'])                 # 4
                 ]  # Swap possible vars from listBaseVars to see if acc improves
    
    
    
    
    
    
    
    
    dictVarRemoved = {'groupRecovery'   :['lnMedianIncome#lnRatePoverty',
                                          'lnProtect#houseMedianValue',
                                          'medianIncome#ratePoverty'
                                          ],   # 
                       'recovery12'     :[], # No interacts to drop
                       'recovery24'     :[], # No interacts to drop
                       'recovery'       :['medianIncome#ratePoverty',
                                          'scoreEcon#scoreStorm',
                                          'lnMedianIncome#lnRatePoverty',
                                          'medianIncome#lnTakeup',
                                          'lnProtect#lnTakeup',
                                          'rateOwnerOcc#sumBuildingDeductib',
                                          'houseMedianValue#sumContentsCoverage'
                                          ]  #
                       } 
    listBaseVars = [x for x in listBaseVars if x not in dictVarRemoved[dv]]
    
    
    
    
    
    
    
    
    if bRunSwap==1 :
        for components,score in listSwaps :
            fltComponents = get_best_forest(dfSubset[dv],
                                            dfSubset[listBaseVars],
                                            listParam,
                                            listIndices
                                            )[1]
            
            fltScore = get_best_forest(dfSubset[dv],
                                       dfSubset[[x for x in listBaseVars if x not in components]+score],
                                       listParam,
                                       listIndices
                                       )[1]
            
            if fltScore>fltComponents : print(str(score)+' is preferred')
            else : print(str(score)+' \tis NOT preferred')
        #   endfor
    #   endif
    



    
    
    
    #   Add score and remove collinear components where necessary
    dictSwapResults = {'groupRecovery':[0,1,3,4], #
                       'recovery12':[0,2,3], 
                       'recovery24':[2,3],
                       'recovery':[0,2,3]
                       } # 0 = Econ, 1 = Insurance, 2 = Coverage, 3 = Deductib, 4 = Housing
    listIV = listBaseVars[:]
    for index in dictSwapResults[dv] :
        components,score = listSwaps[index]
        listIV = [x for x in listIV if x not in components]+score
    #   endfor
    
    
    
    
    
    
    #listIV = listBaseVars[:]
    #   Remove Step-wise identified variables    
    dictVarRemoved = {'groupRecovery'   :['rateOwnerOcc#sumBuildingDeductib',
                                          'scoreStorm',
                                          'sumBuildingCoverage',
                                          'scoreEcon#scoreStorm',
                                          'medianIncome#rateOwnerOcc',
                                          'lnProtect#lnTakeup',
                                          'scoreDeductib',
                                          'scoreInsurance',
                                          'sample_1.0',
                                          'scoreEcon#scoreInsurance',
                                          'scoreStorm#scoreInsurance'
                                          ],   # 
                       'recovery12'     :[], # 
                       'recovery24'     :[], # 
                       'recovery'       :[
                                          ]  #
                       } 
    listIV = [x for x in listIV if x not in dictVarRemoved[dv]]
    
    
    
#                 var       acc
# 5  medianIncome#lnTakeup  0.002754
# 0             sample_2.0  0.000041
# 7              scoreEcon -0.000062
# 1             sample_3.0 -0.000083
# 2              sample_4+ -0.001470
    
    
    
    
    # listIV = listBaseVars[:]
    #   Get base model accuracy
    # fltBaseAccuracy = get_best_forest(dfSubset[dv],
    #                                   dfSubset[listIV],
    #                                   listParam,
    #                                   listIndices
    #                                   )[1]
    
    # dfVarResults = DataFrame(columns=['var','acc']) # df of variable and acc change
    
    # print('\nStarting Backwards Step-wise Algorithm')
    # print('running on list of '+str(listIV))
    # for varOut in listIV :
    #     dictNewRow = {'var':varOut}
        
    #     #print('\nrunning on '+varOut)
        
    #     listKeepVars = listIV[:]
    #     listKeepVars.remove(varOut)
        
    #     fltAcc = get_best_forest(dfSubset[dv],
    #                              dfSubset[listKeepVars],
    #                              listParam,
    #                              listIndices
    #                              )[1]
        
    #     dictNewRow['acc'] = fltAcc
        
    #     dfVarResults = dfVarResults.append(dictNewRow,True)
    #     print('Complete Var: '+varOut+'\t\t\tAcc: '+str(fltAcc))
    # #   endfor
    # dfVarResults['acc'] = dfVarResults['acc'] - fltBaseAccuracy
    # print('\nFor var: '+dv)
    # print('These vars were dropped and led to these increases in accuracy...')
    # print(dfVarResults.nlargest(5,'acc'))
    # for i in range(5) : winsound.Beep(500,500)
    
    
    
    
    
    
    
    
    
    # ####    Multi-Core Approach
    # pool = Pool(cpu_count())
    

    # #dfVarResults = DataFrame(columns=['var','acc']) # df of variable and acc change
    
    # print('\nStarting Backwards Step-wise Algorithm')
    # print('running on list of '+str(listIV))
    # fltBaseAccuracy = get_best_forest(dfSubset[dv],
    #                                 dfSubset[listIV],
    #                                 listParam,
    #                                 listIndices
    #                                 )[1]
    # listMPInput = []
    # for varOut in listIV :
    #     dictNewRow = {'var':varOut}
        
    #     listKeepVars = listIV[:]
    #     listKeepVars.remove(varOut)
        
    #     listMPInput.append((varOut,
    #                        dfSubset[dv],
    #                        dfSubset[listKeepVars],
    #                        listParam,
    #                        listIndices
    #                        ))
    # #   endfor
    # print('Starting multi-processing now...')
    # listResults = pool.map(run_forest_multicore,listMPInput)
    # dfVarResults = DataFrame(listResults,columns=['var','acc'])
    # dfVarResults['acc'] = dfVarResults['acc'] - fltBaseAccuracy
    # print('\nFor var: '+dv)
    # print('These vars were dropped and led to these increases in accuracy...')
    # print(dfVarResults.nlargest(5,'acc'))
    # for i in range(5) : winsound.Beep(500,500)
    # x = input()
    
    
    
    
    
    
    # pool = Pool(cpu_count())    
    # listResult = pool.map(generate_stem,listStem)
    
    
    
    # #   Step-wise find interaction
    
    
    # # get_best_forest(dfSubset[dv],dfSubset[listIV],listParam,listIndices)
    # print('\nStarting Step-Wise Interaction Code')
    # dv = 'recovery'
    # listIV = ['lnTakeup','lnProtect',
    #           'sample_1.0','sample_2.0','sample_3.0','sample_4+',
    #           'sumBuildingCoverage','sumBuildingDeductib',
    #           'sumContentsCoverage','sumContentsDeductib',
    #           'scoreStorm','cat_TD','cat_TS','cat_HU','cat_EX',
    #           'medianIncome','ratePoverty','rateOwnerOcc','houseMedianValue']
    # listParam = [(4,7)]     # Taken from best param for Class Tree on full model above ; used to prevent param+var choices that produce over-fitting
    
    # bTree = 1
    # bDecreasing = 0
    # frame = dfSubset[[dv]+listIV]
    # options.mode.chained_assignment = None  # default='warn'
    
    # loopcount = 0
    
    # # Start while for bDecreasing
    # while bDecreasing != 1 :
    #     dfErr = DataFrame()
        
    #     # Run best-forest for base model
    #     if bTree==1 :
    #         bestMSE = get_best_tree(frame[dv],
    #                                 frame[listIV],
    #                                 listParam,
    #                                 listIndices)[1]
    #     else :
    #         bestMSE = get_best_forest(frame[dv],
    #                                   frame[listIV],
    #                                   listParam,
    #                                   listIndices)[1]
        
    #     # Save var1/var2/MSE for base/base model
    #     dictRow = {'var1':'base','var2':'base','MSE':bestMSE}
    #     dfErr = dfErr.append(dictRow,True)
        
    #     # Get var1 in listIV
    #     for var1 in listIV :
            
    #         # Get var2 in listIV
    #         for var2 in listIV :
                
    #             # Add interaction term
    #             frame[var1+'#'+var2] = frame[var1] * frame[var2]
                
    #             # Run best-forest for base+interation
    #             if bTree==1 :
    #                 bestMSE = get_best_tree(frame[dv],
    #                                         frame[listIV+[var1+'#'+var2]],
    #                                         listParam,
    #                                         listIndices)[1]
    #             else :
    #                 bestMSE = get_best_forest(frame[dv],
    #                                           frame[listIV+[var1+'#'+var2]],
    #                                           listParam,
    #                                           listIndices)[1]
                
    #             # Add var1/var2/MSE for base+interaction
    #             dictRow = {'var1':var1,'var2':var2,'MSE':bestMSE}
    #             dfErr = dfErr.append(dictRow,True)
    #         #   end for
    #     #   end for
    #     # Find var1/var2 for lowest MSE
    #     result1,result2 = dfErr[dfErr['MSE']==dfErr['MSE'].min()].iloc[0][['var1','var2']]
        
    #     # If base/base is best then stop while
    #     loopcount+=1
    #     print('\nEnd of search round : '+str(loopcount))
        
    #     if (result1=='base')&(result2=='base') : 
    #         bDecreasing = 1
    #         print('No additional interaction found.')
        
    #     # Else add var1/var2 to listIV
    #     else :
    #         listIV.append(result1+'#'+result2)
    #         print('New variable added: '+result1+'#'+result2)
    #         # Run KNN w graph
    #     #   end if
    #     # Renew df
    #     frame = frame[[dv]+listIV]
        
    # #   end while
    # # Print listIV
    # print('\nResults: '+str(listIV))
    
    
    
    
#   end main()
# =========================================================================== #






































