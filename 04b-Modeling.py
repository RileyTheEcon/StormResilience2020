# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:21:14 2020
@author: RC

Input files:        data/working/sampleAllCounties.csv
Output files:       

This code fits LASSO, trees and forest for the recovery data.
This is the initial, exploratory set of models.

"""










# =========================================================================== #
from pandas import *
from numpy import log,arange,take,where,std
from sklearn.linear_model import LassoCV,LinearRegression,Lasso
from statsmodels.api import *
from sklearn.neighbors import (NearestNeighbors,KNeighborsClassifier,
                               KNeighborsRegressor,RadiusNeighborsRegressor,
                               NearestCentroid)
from sklearn import tree,metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from statistics import mean
from itertools import product
from scipy.stats import norm


listDVCat = ['recovery','recovery12','recovery24']
listDVNum = ['recovery','lnrecovery','percentScore']


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


# listModel1 = ['s1_EX','s1_TD','s1_TS','s1_H5','s1_H4','s1_H3','s1_H2','s1_H1',
#               's1_SS','s2_LO','s2_TS','s2_EX','s2_H1',
#               's1_vmax','s1_mslp','s1_time','s2_vmax','s2_mslp','s2_time',
#               'takeupTotal','TIVgap','protectGap',
              
#                    ]


                

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
# def storms_to_one (df) :
#     dfStorm = dfSample[(dfSample['bCoastal']==1)&(dfSample['bStorm']==1)
#                    &(dfSample['s1_name'].notnull())]
    
#     # Unique id vars : abrState, county
#     df = DataFrame()
#     for index,row in dfStorm.iterrows():
#         dictRow = {'abrState':row['abrState'],'county':row['county']}
#         if row['numInRecovery']==1:
#             dictRow['vmax'] = row['s1_vmax']
#             dictRow['mslp'] = row['s1_mslp']
#             dictRow['time'] = row['s1_time']
#             dictRow['CAT'] = row['s1_CAT']
#         if row['numInRecovery']==2:
#             dictRow['vmax'] = row['s2_vmax']
#             dictRow['mslp'] = row['s2_mslp']
#             dictRow['time'] = row['s2_time']
#             dictRow['CAT'] = row['s2_CAT']
#         if row['numInRecovery']==3:
#             dictRow['vmax'] = row['s3_vmax']
#             dictRow['mslp'] = row['s3_mslp']
#             dictRow['time'] = row['s3_time']
#             dictRow['CAT'] = row['s3_CAT']
#         df = df.append(dictRow,True)
#     #   end for
#     return df[['abrState','county','vmax','mslp','time','CAT']]
####
# def get_storm_score (df) :
#     for var in ['vmax','mslp','lntime'] :
#         df['Z'+var] = (df[var]-mean(df[var]))/std(df[var])
#     #   end for
#     df['scoreStorm'] = df['Zvmax'] + df['Zmslp'] + df['Zlntime']
#     return df['scoreStorm']
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
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    
    #   Import data & Check distribution of DV
    dfSample = read_csv('data/working/sampleAllCounties.csv')
    # print('\n DV Value Counts')
    # print(dfSample[varDV][~dfSample[['county','s1_name']].duplicated()
    #                       ].value_counts().sort_index())
    
    
    
    
    
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
    
    
    
    
    
    
    
    
    
    



    
    
    # percentScore
    listIV = listStorm + listComped + listPolicy + listCensus + listStormGroup2 + listNumGroup 
    listPercent = [2/len(dfSubset),0.01,0.05]
    listSampleM = [round(x*len(dfSubset)) for x in listPercent]
    
    for dv in listDVCat :
        for intMinSampleSplit in listSampleM :
            # Decision Tree
            modelTree = tree.DecisionTreeClassifier(max_depth=3,min_samples_split=intMinSampleSplit
                                                    ).fit(dfSubset[listIV],dfSubset[dv])
            fig, ax = plt.subplots(figsize=(35,15))
            tree.plot_tree(modelTree,fontsize=15,feature_names=listIV,
                           class_names=True)
            plt.savefig('plot/trees/treedec-'+str(dv)+'-'+str(intMinSampleSplit)+'.png')
        #   end for
    #   end for    
    for dv in listDVNum :
        for intMinSampleSplit in listSampleM :
            # Regression Tree
            modelTree = tree.DecisionTreeRegressor(max_depth=3,min_samples_split=intMinSampleSplit
                                                   ).fit(dfSubset[listIV][dfSubset[dv].notnull()]
                                                         ,dfSubset[dv][dfSubset[dv].notnull()])
            fig, ax = plt.subplots(figsize=(20,20))
            tree.plot_tree(modelTree,fontsize=15,feature_names=listIV,)
            plt.savefig('plot/trees/treereg-'+str(dv)+'-'+str(intMinSampleSplit)+'.png')
            
        #   end for
    #   end for 
    
    
    
    
    
    
    
    
    
    
    
    
    #   Multi-Reg w Tree-chosen variables
    dfOutreg = DataFrame()
    listModels = [('recovery',  ['cat_EX','s1_mslp','houseMedianValue',
                                 's1_vmax','houseTotal','rateOwnerOcc',
                                 'takeupTotal'],
                   'dec-tree'),
                  ('recovery12',['s2_vmax','cat_EX','season_2.0','s1_mslp',
                                 'houseMedianValue','s1_vmax','cat_TD'],
                   'dec-tree'),
                  ('recovery24',['s2_vmax','cat_EX','s1_vmax','s1_mslp',
                                 'houseMedianValue','season_2.0','cat_TS'],
                   'dec-tree'),
                  ('recovery',  ['s2_vmax','houseMedianValue','rateOwnerOcc',
                                 'cat_EX','sumContentsDeductib',
                                 'totalInsurableValue','houseTotal'],
                   'reg-tree'),
                  ('lnrecovery',['s2_vmax','s1_time','season_2.0','s1_mslp',
                                 'policyCount','cat_TS','sumContentsDeductib'],
                   'reg-tree'),
                  ('percentScore',['perCapitaIncome','s1_time','ratePoverty',
                                   'houseMedianValue','rateOwnerOcc'],
                   'reg-tree')
                  ]
    for tpl in listModels :    
        modelMultiReg = OLS(dfSubset[tpl[0]],add_constant(dfSubset[tpl[1]])
                         ).fit()
        dfOutreg = outreg(dfOutreg,modelMultiReg,tpl[0]+'-'+tpl[2],'no clusted std err')
    #   end for
    
    dfOutreg.to_csv('log/tree-to-multireg.csv',index=None)
    
    
    
    
    
    
    
    
    ####    Start ML Model fitting and classification
    strLog = 'Start of Model log:\n'
    listDVCat = ['recovery','recovery12','recovery24']
    listDVNum = ['recovery','lnrecovery','percentScore']
    
    #   Run with takeupTotal if dv!=lnrecovery, else lnTakeup
    #   Run with protectGap if dv!=lnrecovery, else lnProtect
    listVars = ['sample_1.0','sample_2.0','sample_3.0','sample_4+',
                'sumBuildingCoverage','sumBuildingDeductib','sumContentsCoverage','sumContentsDeductib',
                'scoreStorm',
                #'Zlntime','Zvmax','Zmslp',
                'cat_TD','cat_TS','cat_HU','cat_EX',
                # 's1_vmax','s1_mslp','s1_time','s1_TD','s1_TS','s1_HU','s1_EX',
                'medianIncome','ratePoverty','rateOwnerOcc','houseMedianValue'] 
    #s1_LO omitted for multicolinearity # 'month_2.0','season_2.0','season_3.0',
    # takeupTotal
    #   Generate list of train/test indices [(train,test),...]
    listIndices = [(dfTrain,dfTest) for dfTrain,dfTest 
                   in KFold(n_splits=10,shuffle=True).split(dfSubset)]
    
    randSeed = 420
    
    dfDistOut = DataFrame()
    
    
    ####    OLS Regs
    dfOutReg = DataFrame()
    for dv in listDVCat+['lnrecovery','percentScore'] :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
        modelMultiReg = OLS(dfSubset[dv],add_constant(dfSubset[listIV])
                             ).fit()
        dfOutreg = outreg(dfOutreg,modelMultiReg,dv,'no clustered std err')
    #   end for dv
    dfOutreg.to_csv('log/multireg-coefs.csv',index=None)
    
    
    
    
    
    ####    LASSO Model
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
    
    
    
    
    
        
    #   Make param net
    print('\nStart KNN code')
    strLog = strLog+'\nStart KNN code\n'
    listParam = range(2,26)
    ####    KNClassifier
    for dv in listDVCat :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
        dictMSE = {}        
        #   Iter over param net
        for param in listParam : 
            listError = []
            
            #   Iter over train,test set
            for fold in listIndices :
                #   Set train,test
                # trainX = take(dfSubset[listIV],fold[0],axis=0)
                # trainY = take(dfSubset[dv],fold[0],axis=0)
                # testX = take(dfSubset[listIV],fold[1],axis=0)
                # testY = take(dfSubset[dv],fold[1],axis=0)
                
                #   Fit model on train sets
                knnClass = KNeighborsClassifier(n_neighbors=param
                                                ).fit(take(dfSubset[listIV],fold[0],axis=0),
                                                      take(dfSubset[dv],fold[0],axis=0))
                
                #   Calculate error on test set. Save to list
                y_pred = knnClass.predict(take(dfSubset[listIV],fold[1],axis=0))
                listError.append(metrics.accuracy_score(take(dfSubset[dv],fold[1],axis=0)
                                                        ,y_pred))
                
            #   end for fold
            #   Calculate, save MSE for param
            dictMSE[param] = mean(listError)
    
        #   end for param
        #   Report best param and MSE
        bestParam = min(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[bestParam]
        bestKNNClass = KNeighborsClassifier(n_neighbors=bestParam).fit(dfSubset[listIV],
                                                                       dfSubset[dv])
        
        score = bestKNNClass.score(dfSubset[listIV],dfSubset[dv])
        adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
                                               
        if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
        else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
        dfDistOut = get_rating_dist(DataFrame(bestKNNClass.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                    dfSubset[listDistr],
                                    dfDistOut,
                                    'KNN : '+dv,
                                    [0.16,0.5,0.84],
                                    {'mse':str(bestMSE),
                                     'adjR2':str(adjR2),
                                     'alpha':bestParam})
        
        print('For KNN-Classifier, dv='+dv+' k-neighbors='+str(bestParam)+' MSE='
              +str(bestMSE)+' adjR2='+str(adjR2)+'\n')
        strLog = strLog+'For KNN-Classifier, dv='+dv+' k-neighbors='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
    #   end for dv    
    
    
    
    ####    KNCentroid
    for dv in listDVCat :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
        listError = []
        
        #   Iter over train,test set
        for fold in listIndices :
            #   Set train,test
            # trainX = take(dfSubset[listIV],fold[0],axis=0)
            # trainY = take(dfSubset[dv],fold[0],axis=0)
            # testX = take(dfSubset[listIV],fold[1],axis=0)
            # testY = take(dfSubset[dv],fold[1],axis=0)
            
            #   Fit model on train sets
            knnCenter = NearestCentroid().fit(take(dfSubset[listIV],fold[0],axis=0),
                                              take(dfSubset[dv],fold[0],axis=0))
            
            #   Calculate error on test set. Save to list
            y_pred = knnCenter.predict(take(dfSubset[listIV],fold[1],axis=0))
            listError.append(metrics.accuracy_score(take(dfSubset[dv],fold[1],axis=0)
                                                    ,y_pred))
        #   end for fold
        #   Report best param and MSE
        bestMSE = mean(listError)
        
        bestNCenter = NearestCentroid().fit(dfSubset[listIV],dfSubset[dv])
        
        score = bestNCenter.score(dfSubset[listIV],dfSubset[dv])
        adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
        
                                               
        if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
        else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
        dfDistOut = get_rating_dist(DataFrame(bestNCenter.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                    dfSubset[listDistr],
                                    dfDistOut,
                                    'Centroid : '+dv,
                                    [0.16,0.5,0.84],
                                    {'mse':str(bestMSE),
                                     'adjR2':str(adjR2),
                                     'alpha':0})
        
        print('For KNN-Centroid, dv='+dv+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
        strLog = strLog+'For KNN-Centroid, dv='+dv+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
    #   end for dv   
    
    
    
    ####    KNRegressor
    for dv in listDVNum :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
        dictMSE = {}        
        #   Iter over param net
        for param in listParam : 
            listError = []
            
            #   Iter over train,test set
            for fold in listIndices :
                #   Set train,test
                # trainX = take(dfSubset[listIV],fold[0],axis=0)
                # trainY = take(dfSubset[dv],fold[0],axis=0)
                # testX = take(dfSubset[listIV],fold[1],axis=0)
                # testY = take(dfSubset[dv],fold[1],axis=0)
                
                #   Fit model on train sets
                knnReg = KNeighborsRegressor(n_neighbors=param
                                               ).fit(take(dfSubset[listIV],fold[0],axis=0),
                                                      take(dfSubset[dv],fold[0],axis=0))
                
                #   Calculate error on test set. Save to list
                y_pred = knnReg.predict(take(dfSubset[listIV],fold[1],axis=0))
                listError.append(metrics.mean_squared_error(take(dfSubset[dv],fold[1],axis=0)
                                                        ,y_pred))
                
            #   end for fold
            #   Calculate, save MSE for param
            dictMSE[param] = mean(listError)
    
        #   end for param
        #   Report best param and MSE
        bestParam = min(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
        
        bestKNR = KNeighborsRegressor(n_neighbors=bestParam).fit(dfSubset[listIV],
                                                                 dfSubset[dv])
        
        score = bestKNR.score(dfSubset[listIV],dfSubset[dv])
        adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
        
        
        if dv!='lnrecovery' : listDistr = ['lnTakeup','lnProtect','medianIncome','ratePoverty']
        else : listDistr = ['takeupTotal','protectGap','medianIncome','ratePoverty']
        dfDistOut = get_rating_dist(DataFrame(bestKNR.predict(dfSubset[listIV]))[0].apply(to_rating,args=(listRecovery,)),
                                    dfSubset[listDistr],
                                    dfDistOut,
                                    'KNReg : '+dv,
                                    [0.16,0.5,0.84],
                                    {'mse':str(bestMSE),
                                     'adjR2':str(adjR2),
                                     'alpha':bestParam})
        
        print('For KNN-Regressor, dv='+dv+' k-neighbors='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
        strLog = strLog+'For KNN-Classifier, dv='+dv+' k-neighbors='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
    #   end for dv    
    





    
    ####    Trees
    #   Make param list
    listPercent = [2/len(dfSubset),0.01,0.05,0.1]
    listSampleM = [round(x*len(dfSubset)) for x in listPercent]
    listParam = list(product(range(2,9),listSampleM))
    print('\nStart Trees code')
    strLog = strLog+'\nStart Trees code\n'

    ####    Classification Trees
    for dv in listDVCat :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
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
        bestParam = min(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
        bestTreeClass = tree.DecisionTreeClassifier(max_depth=bestParam[0],
                                                  min_samples_split=bestParam[1]
                                                  ).fit(dfSubset[listIV],dfSubset[dv])
        
        score = bestTreeClass.score(dfSubset[listIV],dfSubset[dv])
        adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
        
        fig, ax = plt.subplots(figsize=(100,50))
        tree.plot_tree(bestTreeClass,fontsize=15,feature_names=listIV,)
        plt.savefig('plot/trees/bestClassTree-'+str(dv)+'.png')
        
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
        
        print('For Classification Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
        strLog = strLog+'For Classification Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
    #   end for dv    
    
    
    
    ####    Decision Forest
    for dv in listDVCat :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
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
        bestParam = min(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
        
        bestForestClass = RandomForestClassifier(max_depth=bestParam[0],min_samples_split=bestParam[1]
                                       ).fit(dfSubset[listIV],dfSubset[dv])
        score = bestForestClass.score(dfSubset[listIV],dfSubset[dv])
        adjR2 = 1-(((1-score)*(len(dfSubset)-1))/(len(dfSubset)-len(listIV)-1))
        
        fig, ax = plt.subplots(figsize=(100,50))
        tree.plot_tree(bestForestClass.estimators_[0],fontsize=15,feature_names=listIV,)
        plt.savefig('plot/trees/bestClassForest-'+str(dv)+'.png')
        
        
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
        
        print('For Classification Forest, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
        strLog = strLog+'For Classification Forest, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
    #   end for dv    
    
    
    
    
    ####    Regression Trees
    for dv in listDVNum :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
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
        plt.savefig('plot/trees/bestRegTree-'+str(dv)+'.png')
        
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
        
        print('For Regression Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n')
        strLog = strLog+'For Regression Tree, dv='+dv+' Params='+str(bestParam)+' MSE='+str(bestMSE)+' adjR2='+str(adjR2)+'\n'
    #   end for dv    
    
    
    
    ####    Regression Forest
    for dv in listDVNum :
        if dv=='lnrecovery' : listIV = ['takeupTotal','protectGap']+listVars
        else : listIV = ['lnTakeup','lnProtect']+listVars
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
        plt.savefig('plot/trees/bestRegForest-'+str(dv)+'.png')
        
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
    
    
    print(strLog)
    dfDistOut.to_csv('log/distribution-out.csv',index=None,header=False)
    
    
    
    
    
    
    # #   Fit model on train sets
    # treeClass = tree.DecisionTreeClassifier(max_depth=param[0],min_samples_split=param[1],
    #                                ).fit(take(dfSubset[listIV],fold[0],axis=0),
    #                                       take(dfSubset[dv],fold[0],axis=0))
    
    # #   Calculate error on test set. Save to list
    # y_pred = treeClass.predict(take(dfSubset[listIV],fold[1],axis=0))
    # listError.append(metrics.accuracy_score(take(dfSubset[dv],fold[1],axis=0)
    #                                         ,y_pred))
    
    
    
    
    
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
                listError.append(metrics.mean_squared_error(take(dfY,fold[1],axis=0)
                                                        ,y_pred))
                
            #   end for fold
            #   Calculate, save MSE for param
            dictMSE[param] = mean(listError)
    
        #   end for param
        #   Report best param and MSE
        bestParam = min(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
        return bestParam,bestMSE
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
                listError.append(metrics.mean_squared_error(take(dfY,fold[1],axis=0)
                                                        ,y_pred))
                
            #   end for fold
            #   Calculate, save MSE for param
            dictMSE[param] = mean(listError)
    
        #   end for param
        #   Report best param and MSE
        bestParam = min(dictMSE,key=dictMSE.get)
        bestMSE = dictMSE[min(dictMSE,key=dictMSE.get)]
        return bestParam,bestMSE
    ####
    
    
    
    
    
    #   Step-wise find interaction
    
    
    # get_best_forest(dfSubset[dv],dfSubset[listIV],listParam,listIndices)
    print('\nStarting Step-Wise Interaction Code')
    dv = 'recovery'
    listIV = ['lnTakeup','lnProtect',
              'sample_1.0','sample_2.0','sample_3.0','sample_4+',
              'sumBuildingCoverage','sumBuildingDeductib',
              'sumContentsCoverage','sumContentsDeductib',
              'scoreStorm','cat_TD','cat_TS','cat_HU','cat_EX',
              'medianIncome','ratePoverty','rateOwnerOcc','houseMedianValue']
    listParam = [(4,7)]     # Taken from best param for Class Tree on full model above ; used to prevent param+var choices that produce over-fitting
    
    bTree = 1
    bDecreasing = 0
    frame = dfSubset[[dv]+listIV]
    options.mode.chained_assignment = None  # default='warn'
    
    loopcount = 0
    
    # Start while for bDecreasing
    while bDecreasing != 1 :
        dfErr = DataFrame()
        
        # Run best-forest for base model
        if bTree==1 :
            bestMSE = get_best_tree(frame[dv],
                                    frame[listIV],
                                    listParam,
                                    listIndices)[1]
        else :
            bestMSE = get_best_forest(frame[dv],
                                      frame[listIV],
                                      listParam,
                                      listIndices)[1]
        
        # Save var1/var2/MSE for base/base model
        dictRow = {'var1':'base','var2':'base','MSE':bestMSE}
        dfErr = dfErr.append(dictRow,True)
        
        # Get var1 in listIV
        for var1 in listIV :
            
            # Get var2 in listIV
            for var2 in listIV :
                
                # Add interaction term
                frame[var1+'#'+var2] = frame[var1] * frame[var2]
                
                # Run best-forest for base+interation
                if bTree==1 :
                    bestMSE = get_best_tree(frame[dv],
                                            frame[listIV+[var1+'#'+var2]],
                                            listParam,
                                            listIndices)[1]
                else :
                    bestMSE = get_best_forest(frame[dv],
                                              frame[listIV+[var1+'#'+var2]],
                                              listParam,
                                              listIndices)[1]
                
                # Add var1/var2/MSE for base+interaction
                dictRow = {'var1':var1,'var2':var2,'MSE':bestMSE}
                dfErr = dfErr.append(dictRow,True)
            #   end for
        #   end for
        # Find var1/var2 for lowest MSE
        result1,result2 = dfErr[dfErr['MSE']==dfErr['MSE'].min()].iloc[0][['var1','var2']]
        
        # If base/base is best then stop while
        loopcount+=1
        print('\nEnd of search round : '+str(loopcount))
        
        if (result1=='base')&(result2=='base') : 
            bDecreasing = 1
            print('No additional interaction found.')
        
        # Else add var1/var2 to listIV
        else :
            listIV.append(result1+'#'+result2)
            print('New variable added: '+result1+'#'+result2)
            # Run KNN w graph
        #   end if
        # Renew df
        frame = frame[[dv]+listIV]
        
    #   end while
    # Print listIV
    print('\nResults: '+listIV)
    
    
    # ratePoverty           x   medianIncome
    # sumContentsCoverage   x   houseMedianValue
    # lnTakeup              x   lnProtect
    # medianIncome          x   rateOwnerOcc
    # medianIncome          x   lnTakeup
    # ratePoverty           x   rateOwnerOcc
    

#   end main()
# =========================================================================== #

# dfACS[var] = to_numeric(dfACS[var],errors='coerce')
# dfStem[var] = dfStem[var].fillna(0)

# listStorm  = ['s1_vmax','s1_mslp','s1_time','s2_vmax','s2_mslp','s2_time']
# listComped = ['const','takeupTotal','TIVgap','protectGap']
# listPolicy = ['policyCount','sumBuildingCoverage','sumBuildingDeductib',
#               'sumContentsCoverage','sumContentsDeductib']
# listCensus = ['houseTotal','houseMedianValue','medianIncome','perCapitaIncome',
#               'rateOccupied','rateOwnerOcc','ratePoverty']



# listStormGroup = ['s1_TD','s1_TS','s1_HU','s1_EX','s1_LO',
#                   's2_TD','s2_TS','s2_HU','s2_EX','s2_LO']
# listNumGroup   = ['season_2.0','month_2.0']

# dfPolicy.loc[dfPolicy[replace[0]]==replace[1],replace[0]] = replace[2]


    # Reg-Generated Groups

    
    
    
    # #   Full model w combined hurricae CATs + clustered std.err. 
    # listIV = listStorm + listStormGroup2 + listComped + listPolicy + listCensus + listNumGroup
    # modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
    #                      ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    # dfOutreg = outreg(dfOutreg,modelMultiReg2,'full-model',
    #                   'std. err. clustered by abrState')
    
    
    
    # #   Model minus policy
    # listIV = listStorm + listStormGroup2 + listComped + listCensus + listNumGroup
    # modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
    #                      ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    # dfOutreg = outreg(dfOutreg,modelMultiReg2,'wo-policy',
    #                   'std. err. clustered by abrState')
    
    
    
    # #   Model minus policy + reduced census
    # listIV = listStorm + listStormGroup2 + listComped + ['medianIncome','ratePoverty'] + listNumGroup
    # modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
    #                      ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    # dfOutreg = outreg(dfOutreg,modelMultiReg2,'reduced-census',
    #                   'std. err. clustered by abrState')
    
    
    
    # #   Model minus policy + reduced census + minus CAT
    # listIV = listStorm + listComped + ['medianIncome','ratePoverty'] + listNumGroup
    # modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
    #                      ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    # dfOutreg = outreg(dfOutreg,modelMultiReg2,'wo-CAT',
    #                   'std. err. clustered by abrState')
    
    
    # #   By Comped ; storms
    # for var in listComped :
    #     modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[[var]+['s1_vmax','s1_mslp','s1_time']])
    #                          ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    #     dfOutreg = outreg(dfOutreg,modelMultiReg2,'small-'+var,
    #                       'std. err. clustered by abrState')
    # #   end for
    
    
    
    # #   By Comped ; storms + numIn binaries
    # for var in listComped :
    #     modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[[var]+['s1_vmax','s1_mslp','s1_time']+listNumGroup])
    #                          ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    #     dfOutreg = outreg(dfOutreg,modelMultiReg2,'med-'+var,
    #                       'std. err. clustered by abrState')
    # #   end for
    
    
    
    # #   By Comped ; storms + numIn binaries + reduced census
    # for var in listComped :
    #     modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[[var]+['s1_vmax','s1_mslp','s1_time']+listNumGroup+['medianIncome','ratePoverty']])
    #                          ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    #     dfOutreg = outreg(dfOutreg,modelMultiReg2,'large-'+var,
    #                       'std. err. clustered by abrState')
    # #   end for
    
    
    
    
    
    
    
    # # df = DataFrame()
    # # for var in listComped :
    # #     dictRow = {}
    # #     dictRow['var'] = var
    # #     listIV = listStorm + listStormGroup2 + listCensus + [var]
    # #     modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
    # #                  ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    # #     # print(modelMultiReg.summary())
    # #     dfOutreg = outreg(dfOutreg,modelMultiReg,var,
    # #                       'clustered stderr by abrState')
    # #     dictRow['adj-r2'] = modelMultiReg.rsquared_adj
    # #     dictRow['fstat']  = modelMultiReg.fvalue
    # #     df = df.append(dictRow,True)
    # # #   end for
    # # print(df)
    # dfOutreg.to_csv('log/model-log1.csv',index=None)
    
    
    
    

    
    # # KNN + LASSO
    
    
    # # TREES!
    
    # # All Vars Unrestricted
    # # listIV = listStorm + listComped + listPolicy + listCensus + listStormGroup1 + listNumGroup 
    # # modelTree = tree.DecisionTreeRegressor().fit(dfSubset[listIV],dfSubset[varDV])
    # # fig, ax = plt.subplots(figsize=(100,100))
    # # tree.plot_tree(modelTree,fontsize=10)
    

    # # list(dfSubset[varDV].astype(str).unique())
    
    # dfTreeOut = DataFrame()
    # listRegTrees = [('2', ['s2_vmax','protectGap','rateOccupied',
    #                        'sumContentsDeductib','sumBuildingDeductib',
    #                        'rateOwnerOcc']),
    #                 ('6', ['s2_vmax','protectGap','rateOccupied',
    #                        'rateOwnerOcc']),
    #                 ('32',['s2_vmax','protectGap']),
    #                 ('65',['s2_vmax','protectGap'])
    #                 ]
    # listDecTrees = [('2-6-32',['houseTotal','sumContentsCoverage',
    #                            'takeupTotal','s1_time','s1_mslp',
    #                            'ratePoverty']),
    #                 ('65',    ['houseTotal','sumContentsCoverage',
    #                            'takeupTotal','s1_mslp','ratePoverty'])
    #                 ]
    # for tpl in listRegTrees :    
    #     modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[tpl[1]])
    #                  ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    #     dfTreeOut = outreg(dfTreeOut,modelMultiReg,'reg-tree-'+tpl[0],
    #                       'std. err. clustered by abrState')
    # #   end for
    # for tpl in listDecTrees :
    #     modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[tpl[1]])
    #          ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    #     dfTreeOut = outreg(dfTreeOut,modelMultiReg,'dec-tree'+tpl[0],
    #                       'std. err. clustered by abrState')
    # #   end for

    # listCombined = ['s2_vmax','s1_time','s1_mslp','rateOwnerOcc','rateOccupied',
    #                 'ratePoverty','houseTotal','protectGap','takeupTotal',
    #                 'sumBuildingDeductib','sumContentsDeductib','sumContentsCoverage',]
    # modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[listCombined])
    #                     ).fit(cov_type='cluster', 
    #                           cov_kwds={'groups':dfSubset['abrState']})
    # dfTreeOut = outreg(dfTreeOut,modelMultiReg,'reg-dec-combo',
    #                    'std. err. clustered by abrState')





    # dfTreeOut.to_csv('log/treeregs.csv',index=None)

    # # dfSubset['recovery'][(dfSubset['houseTotal']>18728.5)&
    # #                      (dfSubset['takeupTotal']<=0.036)&
    # #                      (dfSubset['s1_mslp']>1012.5)]
    # # dfVar = DataFrame()
    # # dfVar['vars'] = listIV
    # # print(dfVar)
















