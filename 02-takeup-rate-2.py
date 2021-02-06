# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:21:46 2020
@author: RC

Input files:        data/working/censusACS.csv,
                    data/stem.csv,
                    data/censusMaster.csv,
                    data/working/policy+claims2.csv,
Preceeding Script:  
Output files:       data/working/takeupMonthly.csv,     # Used for sample gen
                    data/working/takeupYearly.csv       # Used for reporting
Following Script:   

This is the CURRENT version of this file.
This takes the prepared data from Production and the American Community
Survey and combines them to produce takeup rates.

"""










# =========================================================================== #
from pandas import *
from multiprocessing import *
#   Takes: 'mean','median','max','dec','sum'
strMethod = 'mean'
bGenStem = 0    #Run stem generate code--must be run in console window
bAddMissingYears = 0
listKeep = ['houseTotalUnits',
            'houseOwnerOccupied',
            'econPoverty',
            'houseMedianValue',
            'econMedianIncome',
            'econPerCapitaIncome',
            'econTotalHouseholds'
            ] # list of vars from ACS to keep
# =========================================================================== #










# =========================================================================== #
def merge_track (left,right,by,how='outer',overwrite=0) :
    #   Fixed version!
    left['fromSource'] = 1
    right['fromMerger'] = 1
    
    if (how=='left')&(len(right[right[by].duplicated()])>0)&(overwrite==0) :
        print('Merger not unique')
        raise SystemExit(0)
    else : dfMerge = left.merge(right,how=how,on=by)
    
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
def fix_dupe_county (df,groupby) :
    dfSingle = df[~df[groupby].duplicated(keep=False)]
    dfDupe = df[df[groupby].duplicated(keep=False)]
    
    dfUnique = dfDupe[groupby].drop_duplicates(subset=groupby)
    dfUnique = dfUnique.merge(dfDupe.groupby(groupby)[['houseTotalUnits','houseOwnerOccupied'
                                                       ]].sum().reset_index(),how='left',on=groupby)
    dfUnique = dfUnique.merge(dfDupe.groupby(groupby)[['econPoverty','houseMedianValue',
                                                       'econMedianIncome','econPerCapitaIncome',
                                                       'econTotalHouseholds'
                                                       ]].mean().reset_index(),how='left',on=groupby)
    
    # dfDupe = dfDupe.groupby(groupby).sum().reset_index()
    
    df = dfSingle.append(dfUnique,ignore_index=True)
    
    return df
####
def remove_vars (df,listVar) :
    for var in listVar :
        del df[var]
    return df
####
def month_to_year (df,method) :
    if method == 'mean' :
        df = df.groupby(['year','state','county'])['policyTotalBuildingInsuranceCoverage',
                                                   'policyPolicyCount'].mean().reset_index()
        #df = df.rename(columns={'policyPolicyCount':'meanPolicyCount'})
    ####
    return df
####
def add_month_rows (df) :
    pool = Pool(cpu_count())
    dfNotNull = dfMerge[dfMerge['fromSource'].notnull()]
    dfIsNa = dfMerge[dfMerge['fromSource'].isna()]
    
    def add_twelve (dictRow) :
        df = DataFrame()
        for x in range(12) :
            dictRow['month'] = x+1
            df = df.append(dictRow,True)
        #   end for
    ####
            
    total = len(dfIsNa)
    count = 1
    listRow = []
    
    for index,row in dfIsNa.iterrows() :
        listRow.append(dict(row))
    #   end for
    listResult = pool.map(add_twelve,listRow)    
    dfNotNull = dfNotNull.append(concat(listResult))
    
    return dfNotNull
####
def collapse_rows (df,listGroup,listMean=[],listSum=[],listConcat=[]) :
    # Tree
    dfTrunk = df[listGroup].drop_duplicates(subset=listGroup)
    
    # Branches
    # Mean
    if len(listMean)>0 :
        dfTrunk = dfTrunk.merge(df.groupby(listGroup)[listMean].mean().reset_index(),
                                how='left',on=listGroup)
    # Sum
    if len(listSum)>0 :
        dfTrunk = dfTrunk.merge(df.groupby(listGroup)[listSum].sum().reset_index(),
                                how='left',on=listGroup)
    # Concat -- still need
    return dfTrunk
####
def generate_stem (tpl) :
    # tpl = (year,state,county)
    df = DataFrame()
    dictRow = {'year':tpl[0],'state':tpl[1],'county':tpl[2]}
    for x in range(1,13) :
        dictRow['month'] = x
        df = df.append(dictRow,True)
    #   end for
    return df
####
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    print(__doc__)
    
    ####    Import American Community Survey
    dfACS       = read_csv('data/working/censusACS.csv')
    
    ####    Subset df / correct for 
    dfACS = dfACS[dfACS['state']!='Puerto Rico']
    dfACS = dfACS[['year','county','state']+listKeep]
    for var in listKeep :
        dfACS[var] = to_numeric(dfACS[var],errors='coerce')
    #   for end
    dfACS = fix_dupe_county(dfACS,['year','state','county'])
    
    








    #   Generate Stem from ACS
    # Create list of unique year, state, county combos
    if bGenStem == 1 :
        df = dfACS[['year','state','county']]
        df = df.drop_duplicates()
        listStem = list(df.itertuples(index=False,name=None))
        
        # Take list and convert to complete list with 12 months for each year
        print('generating stem')
        pool = Pool(cpu_count())    
        listResult = pool.map(generate_stem,listStem)
        dfStem = concat(listResult)
        print('exporting stem')
        dfStem.to_csv('data/stem.csv',index=None)
    else : dfStem = read_csv('data/stem.csv')
    
    
    
    
    
    
    
    
    
    
    #   Census Master Data to connect leafs to stem
    # Import
    dfCensus    = read_csv('data/censusMaster.csv')
    
    # Gen FIPS list
    dfFIPS = dfCensus.drop_duplicates(subset=['fips'])
    dfFIPS = dfFIPS[['fips','county','state']]
    
    # Gen Counties list
    dfCounty = dfCensus.drop_duplicates(subset=['county','state'])
    dfCounty = dfCounty[['county','state']]
    
    # Add abrState to stem
    #dfStem = dfStem.merge(dfCensus[['state','abrState']],how='left',on='state')
    
    # Add dfACS to stem
    dfStem = dfStem.merge(dfACS,how='left',on=['year','state','county'])
    
    
      
    
    
    
    
    
    
    
    #   Policies
    # Import
    dfPolicy    = read_csv('data/working/policy+claims2.csv')
    
    # Drop cols and rows
    dfPolicy = dfPolicy[['year','month','fips','policyPolicyCount',
                         'policyTotalBuildingInsuranceCoverage',
                         'policyTotalContentsInsuranceCoverage',
                         'policyDeductibleAmountinBuildingCoverage',
                         'policyDeductibleAmountinContentsCoverage',
                         ]]
    dfPolicy = dfPolicy[(2010<=dfPolicy['year'])&(dfPolicy['year']<=2020)]
    
    
    
    
    
    
    
    
    
    
    # Add census master data
    dfPolicy = merge_track(dfPolicy,dfFIPS,'fips','left')[0]
    dfPolicy = remove_vars(dfPolicy,['fromSource','fromMerger','MergeSuccess'])
    
    # Drop territories
    dfPolicy = dfPolicy[(dfPolicy['state']!='American Samoa')&
                (dfPolicy['state']!='Mariana Islands')&
                (dfPolicy['state']!='Puerto Rico')&
                (dfPolicy['state']!='Guam')&
                (dfPolicy['state']!='Virgin Islands')]

    # Collapse fips up to counties
    dfPolicy = dfPolicy.groupby(['year','month','state','county'
                                 ])[['policyTotalBuildingInsuranceCoverage',
                                     'policyTotalContentsInsuranceCoverage',
                                     'policyDeductibleAmountinBuildingCoverage',
                                     'policyDeductibleAmountinContentsCoverage',
                                     'policyPolicyCount']].sum().reset_index()
    


    
    
    
                                     
                                     
                                     
                                     
    #   Replace incomplete years 2010, 2019
    if bAddMissingYears==1 :
        dfPolicy = dfPolicy[(dfPolicy['year']>2010)&(dfPolicy['year']<2019)]
        df2010 = dfPolicy[dfPolicy['year']==2011]
        df2010['year'] = 2010
        df2019 = dfPolicy[dfPolicy['year']==2018]
        df2019['year'] = 2019
        dfPolicy = concat([df2010,dfPolicy,df2019])
    #   endif
    
    

    
    # Add Policy data to Stem
    dfStem = dfStem.merge(dfPolicy,how='left',on=['year','month',
                                                  'state','county'])





    
    
    
    
    
    #   With completed stem-leafs, rename vars, calculate, export
    # Vars in stem :    year, month, state, abrState, county, 
    #                   'houseTotalUnits','houseOwnerOccupied',
    #                   'econPoverty','houseMedianValue',
    #                   'econMedianIncome','econPerCapitaIncome',
    #                   econTotalHouseHolds,
    #                   policyTotalBuildingInsuranceCoverage,
    #                   policyPolicyCount,
    dfStem = dfStem.rename(columns={'policyPolicyCount':'policyCount',
                                    'policyTotalBuildingInsuranceCoverage':'sumBuildingCoverage',
                                    'policyTotalContentsInsuranceCoverage':'sumContentsCoverage',
                                    'policyDeductibleAmountinBuildingCoverage':'sumBuildingDeductib',
                                    'policyDeductibleAmountinContentsCoverage':'sumContentsDeductib',
                                    'econTotalHouseholds':'houseOccupied',
                                    'econPerCapitaIncome':'perCapitaIncome',
                                    'econMedianIncome':'medianIncome',
                                    #'houseMedianValue',
                                    'econPoverty':'ratePoverty',
                                    #'houseOwnerOccupied',
                                    'houseTotalUnits':'houseTotal'
                                    })
    
    for var in ['sumBuildingCoverage','sumContentsCoverage',
                'sumBuildingDeductib','sumContentsDeductib','policyCount'] :
        dfStem[var] = dfStem[var].fillna(0)
    #   endfor
    
    
    
    
    
    
    
    
    
    
    # Set yearly data aside
    dfYear = collapse_rows(dfStem,
                       listGroup=['year','state','county'],
                       listMean=['policyCount','sumBuildingCoverage',
                                 'houseOccupied','perCapitaIncome',
                                 'houseMedianValue','medianIncome',
                                 'ratePoverty','houseOwnerOccupied',
                                 'houseTotal','sumContentsCoverage',
                                 'sumBuildingDeductib','sumContentsDeductib'])
    
    
    
    
    
    
    
    
    
    
    # Make calculations
    dfStem['takeupTotal'] = dfStem['policyCount'] / dfStem['houseTotal']
    dfStem['rateOccupied'] = dfStem['houseOccupied'] / dfStem['houseTotal']
    dfStem['rateOwnerOcc'] = dfStem['houseOwnerOccupied'] / dfStem['houseTotal']
    dfStem['totalInsurableValue'] = 1.4*dfStem['houseTotal']*dfStem['houseMedianValue']
    dfStem['insuredValue'] = (dfStem['sumBuildingCoverage'] 
                              + dfStem['sumContentsCoverage']
                              - dfStem['sumBuildingDeductib'] 
                              - dfStem['sumContentsDeductib'])
    dfStem['protectGap'] = dfStem['totalInsurableValue'] - dfStem['insuredValue']
    
    
        
    
    
    
    
    
    
    
    dfStem = dfStem[['year','month','state','county',
                     'takeupTotal',
                     'protectGap',
                     'totalInsurableValue',
                     'insuredValue',
                     'rateOccupied','rateOwnerOcc',
                     'perCapitaIncome','medianIncome','houseMedianValue',
                     'policyCount','ratePoverty','houseTotal',
                     'sumBuildingCoverage',
                     'sumContentsCoverage',
                     'sumBuildingDeductib',
                     'sumContentsDeductib'
                     ]]
    
    dfStem.to_csv('data/working/takeupMonthly.csv',index=None)




    





    #   Gen vars
    dfYear['takeupTotal'] = dfYear['policyCount'] / dfYear['houseTotal']
    dfYear['rateOccupied'] = dfYear['houseOccupied'] / dfYear['houseTotal']
    dfYear['rateOwnerOcc'] = dfYear['houseOwnerOccupied'] / dfYear['houseTotal']
    dfYear['totalInsurableValue'] = 1.4*dfYear['houseTotal']*dfYear['houseMedianValue']
    dfYear['insuredValue'] = (dfYear['sumBuildingCoverage'] 
                              + dfYear['sumContentsCoverage']
                              - dfYear['sumBuildingDeductib'] 
                              - dfYear['sumContentsDeductib'])
    dfYear['protectGap'] = dfYear['totalInsurableValue'] - dfYear['insuredValue']
    
    
    
    
    
    
    
    
    
    
    #   Get 2020 as an average
    dfYear = dfYear[dfYear['year']<2020]
    df2020 = dfYear.groupby(['state','county']).mean().reset_index()
    df2020['year'] = 2020
    dfYear = dfYear.append(df2020)
    print(dfYear.groupby('year')['takeupTotal'].mean())
    
    
    
    
    
    
    
    
    
    
    dfYear = dfYear[['year','state','county',
                     'takeupTotal',
                     'protectGap',
                     'totalInsurableValue',
                     'insuredValue',
                     'rateOccupied','rateOwnerOcc',
                     'perCapitaIncome','medianIncome','houseMedianValue',
                     'policyCount','ratePoverty','houseTotal',
                     'sumBuildingCoverage',
                     'sumContentsCoverage',
                     'sumBuildingDeductib',
                     'sumContentsDeductib'
                     ]]
    
    dfYear.to_csv('data/working/takeupYearly.csv',index=None)
#   endmain
# =========================================================================== #










































