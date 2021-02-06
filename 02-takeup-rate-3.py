# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:57:15 2020
@author: RC

Input files:        data/working/censusACS.csv,
                    data/stem.csv,
                    data/censusMaster.csv,
                    data/working/policy+claims2.csv,
Preceeding Script:  
Output files:       data/working/takeupMonthly2.csv,     # Used for sample gen
                    data/working/takeupYearly2.csv       # Used for reporting
Following Script:   

This is the DEPRECATED version of this file.
This takes the prepared data from Production and the American Community
Survey and combines them to produce takeup rates. This was made for producing
results for reporting with fewer variables.

"""










# =========================================================================== #
from pandas import *
from multiprocessing import *
#   Takes: 'mean','median','max','dec','sum'
strMethod = 'mean'

listKeep = ['houseTotalUnits',
            'houseMedianValue'
            ] #list of vars from ACS to keep
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
    dfUnique = dfUnique.merge(dfDupe.groupby(groupby)[['houseTotalUnits',
                                                       #'houseOwnerOccupied'
                                                       ]].sum().reset_index(),how='left',on=groupby)
    dfUnique = dfUnique.merge(dfDupe.groupby(groupby)[['houseMedianValue',
                                                       #'econPoverty',
                                                       #'econMedianIncome',
                                                       #'econPerCapitaIncome',
                                                       #'econTotalHouseholds'
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
    #   ACS Data
    # Import Amer Comm Surv data
    print('importing ACS')
    dfACS       = read_csv('data/working/censusACS.csv')
    
    # Drop Puerto Rico & Select columns
    dfACS = dfACS[dfACS['state']!='Puerto Rico']
    dfACS = dfACS[['year','county','state']+listKeep]
    
    # Convert str nums to num nums
    for var in listKeep :
        dfACS[var] = to_numeric(dfACS[var],errors='coerce')
    #   for end
    dfACS = fix_dupe_county(dfACS,['year','state','county'])
    
    



    




    #   Generate Stem from ACS if it does not already exist
    # Create list of unique year, state, county combos
    bGenStem = 0
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
    print('importing census key')
    dfCensus    = read_csv('data/censusMaster.csv')
    
    # Gen FIPS list
    dfFIPS = dfCensus.drop_duplicates(subset=['fips'])
    dfFIPS = dfFIPS[['fips','county','state']]
    
    # Gen Counties list
    dfCounty = dfCensus.drop_duplicates(subset=['county','state'])
    dfCounty = dfCounty[['county','state']]
    
    # Get state numbers
    dfState = dfCensus.drop_duplicates(subset=['state','stateCode'])
    dfState = dfState[['state','stateCode']]
    
    # Add census data to unique id stem
    dfStem = dfStem.merge(dfACS,how='left',on=['year','state','county'])
    
    
    
    
 
    
    
    
    
    
    #   Policies
    # Import 2018/2019 policy data
    dfPolicy    = read_csv('data/working/policy+claims2.csv')
    
    # Drop cols and rows
    dfPolicy = dfPolicy[['year','month','fips','policyPolicyCount',
                         'policyTotalBuildingInsuranceCoverage',
                         'policyTotalContentsInsuranceCoverage',
                         'policyDeductibleAmountinBuildingCoverage',
                         'policyDeductibleAmountinContentsCoverage',
                         'policyTotalInsurancePremiumofthePolicy',
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

    # Collapse fips up to counties--add individual quantities to county-level
    dfPolicy = dfPolicy.groupby(['year','month','state','county'
                                 ])[['policyTotalBuildingInsuranceCoverage',
                                     'policyTotalContentsInsuranceCoverage',
                                     'policyDeductibleAmountinBuildingCoverage',
                                     'policyDeductibleAmountinContentsCoverage',
                                     'policyTotalInsurancePremiumofthePolicy',
                                     'policyPolicyCount']].sum().reset_index()
    
    # Add Policy data to unique id stem
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
                                    'houseTotalUnits':'houseTotal',
                                    'policyTotalInsurancePremiumofthePolicy':'sumPolicyPremium',
                                    })
    
    for var in ['sumBuildingCoverage','sumContentsCoverage',
                'sumBuildingDeductib','sumContentsDeductib',
                'sumPolicyPremium','policyCount'] :
        dfStem[var] = dfStem[var].fillna(0)
    ####
    
    # Set yearly data aside
    dfYear = collapse_rows(dfStem,
                       listGroup=['year','state','county'],
                       listMean=['policyCount','houseTotal','houseMedianValue',
                                 'sumBuildingCoverage','sumContentsCoverage',
                                 'sumBuildingDeductib','sumContentsDeductib',
                                 'sumPolicyPremium'])
    
    
    
    
    
    
    
    
    
    
    # Make calculations
    dfStem['takeupTotal'] = dfStem['policyCount'] / dfStem['houseTotal']
    dfStem['protectGap'] = (dfStem['houseTotal']*dfStem['houseMedianValue'])-(dfStem['policyCount']*350000)
    dfStem['TIVgap'] = (dfStem['sumBuildingCoverage'] + dfStem['sumContentsCoverage']
                        - dfStem['sumBuildingDeductib'] - dfStem['sumContentsDeductib'])
    
    
    
    
    
    
    
    
    
    
    dfStem = dfStem.merge(dfState,how='left',on=['state'])
    
    dfStem = dfStem[['year','month','state','stateCode','county',
                     'policyCount','houseTotal','TIVgap','takeupTotal',
                     'protectGap','sumPolicyPremium']]
    
    dfStem.to_csv('data/working/takeupMonthly2.csv',index=None)
    
    
    
    
    
    
    
    
    
    
    dfYear['takeupTotal'] = dfYear['policyCount'] / dfYear['houseTotal']
    dfYear['protectGap'] = (dfYear['houseTotal']*dfYear['houseMedianValue'])-(dfYear['policyCount']*350000)
    dfYear['TIVgap'] = (dfYear['sumBuildingCoverage'] + dfYear['sumContentsCoverage']
                        - dfYear['sumBuildingDeductib'] - dfYear['sumContentsDeductib'])
    
    
    
    
    
    
    
    
    
    
    dfYear = dfYear.merge(dfState,how='left',on=['state'])
    
    dfYear = dfYear[['year','state','stateCode','county',
                     'policyCount','houseTotal','TIVgap','takeupTotal',
                     'protectGap','sumPolicyPremium']]
    
    dfYear.to_csv('data/working/takeupYearly2.csv',index=None)
    
#   endmain
# =========================================================================== #

# 'claimsPolicyCount',
# 'claimsAmountPaidonBuildingClaim',
# 'claimsAmountPaidonContentsClaim',
# 'claimsTotalBuildingInsuranceCoverage',
# 'claimsTotalContentsInsuranceCoverage',


































