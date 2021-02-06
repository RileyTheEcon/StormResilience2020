# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:21:14 2020
@author: RC

Input files:        data/working/sampleAllCounties.csv
Output files:       

This code fits mult-reg OLS, LASSO, trees and forest for the recovery data.
This is the initial, exploratory set of models.

"""










# =========================================================================== #
from pandas import *
from sklearn.linear_model import LinearRegression
from statsmodels.api import *
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
import matplotlib.pyplot as plt




varDV = 'recovery'
listNotNeeded = ['year','month','county','abrState','prevProd','Production',
                 'numInMonth','numInSeason','recoverCount',
                 's1_name','s1_CAT','s2_name','s2_CAT']
listCategorical = [('s1_CAT','s1_'),('s2_CAT','s2_'),
                   ('numInSeason','season_'),('numInMonth','month_')]

listStorm  = ['s1_vmax','s1_mslp','s1_time','s2_vmax','s2_mslp','s2_time']
listComped = ['takeupTotal','TIVgap','protectGap']
listPolicy = ['policyCount','sumBuildingCoverage','sumBuildingDeductib',
              'sumContentsCoverage','sumContentsDeductib']
listCensus = ['houseTotal','houseMedianValue','medianIncome','perCapitaIncome',
              'rateOccupied','rateOwnerOcc','ratePoverty']


# listModel1 = ['s1_EX','s1_TD','s1_TS','s1_H5','s1_H4','s1_H3','s1_H2','s1_H1',
#               's1_SS','s2_LO','s2_TS','s2_EX','s2_H1',
#               's1_vmax','s1_mslp','s1_time','s2_vmax','s2_mslp','s2_time',
#               'takeupTotal','TIVgap','protectGap',
              
#                    ]


listStormGroup1 = ['s1_EX','s1_TD','s1_TS','s1_H5','s1_H4','s1_H3','s1_H2','s1_H1',
                   's1_SS','s2_LO','s2_TS','s2_EX','s2_H1',]     # omitted s1_LO , s2_TD
listStormGroup2 = ['s1_TD','s1_TS','s1_HU','s1_EX','s1_LO',
                   's2_TD','s2_TS','s2_HU','s2_EX','s2_LO']
listNumGroup    = ['season_2.0','month_2.0']    #omitted season_1.0 , month_1.0
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
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    print(__doc__)
    dfOutreg = DataFrame()
    
    #   Import data & Check distribution of DV
    dfSample = read_csv('data/working/sampleAllCounties.csv')
    # print('\n DV Value Counts')
    # print(dfSample[varDV][~dfSample[['county','s1_name']].duplicated()
    #                       ].value_counts().sort_index())
    
    
    
    
    
    
    
    
    
    
    #   Catgories to indicator
    # Get indicators for CAT
    # [(var name,prefix)]
    listDrop = []           # Need to remove for perfect multicollinearity
    strMultiCol = 'Dropped for multi-collinearity : '
    for category in listCategorical :    
        print('\n Value Counts for '+str(category[0]))
        print(get_dummies(dfSample[category[0]]).sum())
        intFirst = 1
        for entry in dfSample[category[0]][dfSample[category[0]].notnull()].unique() :
            dfSample[category[1]+str(entry)] = dfSample[category[0]] == entry
            dfSample[category[1]+str(entry)] = dfSample[category[1]+str(entry)].astype(int)
        #   end for
    #   end for
    
    
    
    
    
    
    
    
    
    
    #   Make Subset
    # Make subset for Only effected counties + only effected months
    dfSubset = dfSample[(dfSample['bEffectedCounty']==1)&
                        (dfSample['bStorm']==1)&
                        (dfSample['bCoastal']==1)]
    for var in ['prevProd','Production','s2_vmax','s2_mslp','s2_time'] :
        dfSubset[var] = to_numeric(dfSubset[var],errors='coerce')
        dfSubset[var] = dfSubset[var].fillna(0)
    #   end for
    
    print('\n DV Value Counts')
    print(dfSubset['recovery'].value_counts().sort_index())
    
    print('\nCount of Recovery in [1,2) Year Recovery Time :')
    print(dfSubset['recovery'][(dfSubset['recovery']>12)&(dfSubset['recovery']<25)].value_counts().sort_index())
    
    print('\nMean Num in Recovery for [1,2) Year Recovery Time :')
    print(dfSubset[(dfSubset['recovery']>12)&(dfSubset['recovery']<25)].groupby(['recovery'])['numInSeason'].mean())
    
    
    
    
    
    
    
    
    
    
    #   Multi-Reg + Step-wise Reg
    listIV = listStorm + listComped + listPolicy + listCensus + listStormGroup1 + listNumGroup
    modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])).fit()
    # dfOutreg = outreg(dfOutreg,modelMultiReg,'fullmodel','')

    modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
                        ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    # print(modelMultiReg.summary())
    # dfOutreg = outreg(dfOutreg,modelMultiReg,'fullmodel2',
    #                   'clustered stderr by abrState')



    
    
    
    
    
    

    # Reg-Generated Groups
    listHurr = ['H1','H2','H3','H4','H5']
    listTS   = ['TS','SS']
    
    dfSubset['s1_HU'] = dfSample['s1_CAT'].apply(lambda x : 1 if x in listHurr else 0)
    dfSubset['s1_TS'] = dfSample['s1_CAT'].apply(lambda x : 1 if x in listTS else 0)
    dfSubset['s2_HU'] = dfSample['s2_CAT'].apply(lambda x : 1 if x in listHurr else 0)
    dfSubset['s2_TS'] = dfSample['s2_CAT'].apply(lambda x : 1 if x in listTS else 0)
    
    
    
    
    
    
    
    
    
    
    #   Full model w combined hurricae CATs + clustered std.err. 
    listIV = listStorm + listStormGroup2 + listComped + listPolicy + listCensus + listNumGroup
    modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
                         ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    dfOutreg = outreg(dfOutreg,modelMultiReg2,'full-model',
                      'std. err. clustered by abrState')
    
    
    
    
    
    
    
    
    
    
    #   Model minus policy
    listIV = listStorm + listStormGroup2 + listComped + listCensus + listNumGroup
    modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
                         ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    dfOutreg = outreg(dfOutreg,modelMultiReg2,'wo-policy',
                      'std. err. clustered by abrState')
    
    
    
    
    
    
    
    
    
    
    #   Model minus policy + reduced census
    listIV = listStorm + listStormGroup2 + listComped + ['medianIncome','ratePoverty'] + listNumGroup
    modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
                         ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    dfOutreg = outreg(dfOutreg,modelMultiReg2,'reduced-census',
                      'std. err. clustered by abrState')
    
    
    
    
    
    
    
    
    
    
    #   Model minus policy + reduced census + minus CAT
    listIV = listStorm + listComped + ['medianIncome','ratePoverty'] + listNumGroup
    modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
                         ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    dfOutreg = outreg(dfOutreg,modelMultiReg2,'wo-CAT',
                      'std. err. clustered by abrState')
    
    
    
    
    
    
    
    
    
    
    #   By Comped ; storms
    for var in listComped :
        modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[[var]+['s1_vmax','s1_mslp','s1_time']])
                             ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
        dfOutreg = outreg(dfOutreg,modelMultiReg2,'small-'+var,
                          'std. err. clustered by abrState')
    #   end for
    
    
    
    
    
    
    
    
    
    
    #   By Comped ; storms + numIn binaries
    for var in listComped :
        modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[[var]+['s1_vmax','s1_mslp','s1_time']+listNumGroup])
                             ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
        dfOutreg = outreg(dfOutreg,modelMultiReg2,'med-'+var,
                          'std. err. clustered by abrState')
    #   end for
    
    
    
     
    
    
    
    
    
    
    #   By Comped ; storms + numIn binaries + reduced census
    for var in listComped :
        modelMultiReg2 = OLS(dfSubset[varDV],add_constant(dfSubset[[var]+['s1_vmax','s1_mslp','s1_time']+listNumGroup+['medianIncome','ratePoverty']])
                             ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
        dfOutreg = outreg(dfOutreg,modelMultiReg2,'large-'+var,
                          'std. err. clustered by abrState')
    #   end for
    
    
    
    
    
    
    
    
    
    
    # df = DataFrame()
    # for var in listComped :
    #     dictRow = {}
    #     dictRow['var'] = var
    #     listIV = listStorm + listStormGroup2 + listCensus + [var]
    #     modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[listIV])
    #                  ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
    #     # print(modelMultiReg.summary())
    #     dfOutreg = outreg(dfOutreg,modelMultiReg,var,
    #                       'clustered stderr by abrState')
    #     dictRow['adj-r2'] = modelMultiReg.rsquared_adj
    #     dictRow['fstat']  = modelMultiReg.fvalue
    #     df = df.append(dictRow,True)
    # #   end for
    # print(df)
    dfOutreg.to_csv('log/model-log1.csv',index=None)
    
    
    
    

    
    
    
    
    
    # TREES!
    # All Vars Unrestricted
    # listIV = listStorm + listComped + listPolicy + listCensus + listStormGroup1 + listNumGroup 
    # modelTree = tree.DecisionTreeRegressor().fit(dfSubset[listIV],dfSubset[varDV])
    # fig, ax = plt.subplots(figsize=(100,100))
    # tree.plot_tree(modelTree,fontsize=10)
    listIV = listStorm + listComped + listPolicy + listCensus + listStormGroup1 + listNumGroup 
    listPercent = [2/len(dfSubset),0.01,0.05,0.1]
    
    for percent in listPercent :
        intMinSampleSplit = round(percent * len(dfSubset))
        # Regression Tree
        modelTree = tree.DecisionTreeRegressor(max_depth=3,min_samples_split=intMinSampleSplit
                                               ).fit(dfSubset[listIV],dfSubset[varDV])
        fig, ax = plt.subplots(figsize=(20,20))
        tree.plot_tree(modelTree,fontsize=15,feature_names=listIV,)
        plt.savefig('plot/trees/reg-tree-min-'+str(intMinSampleSplit)+'.png')
        
        # Decision Tree
        modelTree = tree.DecisionTreeClassifier(max_depth=3,min_samples_split=intMinSampleSplit
                                                ).fit(dfSubset[listIV],dfSubset[varDV])
        fig, ax = plt.subplots(figsize=(35,15))
        tree.plot_tree(modelTree,fontsize=15,feature_names=listIV,
                       class_names=True)
        plt.savefig('plot/trees/dec-tree-min-'+str(intMinSampleSplit)+'.png')
    #   end for
    
    
    
    
    
    
    
    
    
    
    dfTreeOut = DataFrame()
    listRegTrees = [('2', ['s2_vmax','protectGap','rateOccupied',
                           'sumContentsDeductib','sumBuildingDeductib',
                           'rateOwnerOcc']),
                    ('6', ['s2_vmax','protectGap','rateOccupied',
                           'rateOwnerOcc']),
                    ('32',['s2_vmax','protectGap']),
                    ('65',['s2_vmax','protectGap'])
                    ]
    listDecTrees = [('2-6-32',['houseTotal','sumContentsCoverage',
                               'takeupTotal','s1_time','s1_mslp',
                               'ratePoverty']),
                    ('65',    ['houseTotal','sumContentsCoverage',
                               'takeupTotal','s1_mslp','ratePoverty'])
                    ]
    for tpl in listRegTrees :    
        modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[tpl[1]])
                     ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
        dfTreeOut = outreg(dfTreeOut,modelMultiReg,'reg-tree-'+tpl[0],
                          'std. err. clustered by abrState')
    #   end for
    for tpl in listDecTrees :
        modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[tpl[1]])
             ).fit(cov_type='cluster', cov_kwds={'groups':dfSubset['abrState']})
        dfTreeOut = outreg(dfTreeOut,modelMultiReg,'dec-tree'+tpl[0],
                          'std. err. clustered by abrState')
    #   end for










    listCombined = ['s2_vmax','s1_time','s1_mslp','rateOwnerOcc','rateOccupied',
                    'ratePoverty','houseTotal','protectGap','takeupTotal',
                    'sumBuildingDeductib','sumContentsDeductib','sumContentsCoverage',]
    modelMultiReg = OLS(dfSubset[varDV],add_constant(dfSubset[listCombined])
                        ).fit(cov_type='cluster', 
                              cov_kwds={'groups':dfSubset['abrState']})
    dfTreeOut = outreg(dfTreeOut,modelMultiReg,'reg-dec-combo',
                       'std. err. clustered by abrState')










    dfTreeOut.to_csv('log/treeregs.csv',index=None)

#   end main()
# =========================================================================== #

    # dfSubset['recovery'][(dfSubset['houseTotal']>18728.5)&
    #                      (dfSubset['takeupTotal']<=0.036)&
    #                      (dfSubset['s1_mslp']>1012.5)]
    # dfVar = DataFrame()
    # dfVar['vars'] = listIV
    # print(dfVar)

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



















