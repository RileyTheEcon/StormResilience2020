# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:15:24 2020
@author: RC

Input files:        data/census FIPS Codes.txt,
                    data/ZIP-COUNTY-FIPS_2018-03.csv,
Preceeding Script:  none
Output files:       data/censusMaster.csv
Following Script:   any that use censusMaster dictionary

We take the separate lists for location data and create a single directory for
states/fips/zips/counties.

"""

# =============================================================================
from pandas import *
# =============================================================================

# =============================================================================
dictZipFips = {'ZIP':'zip',
               'CITY':'city',
               'STATE':'abrState',
               'COUNTYNAME':'county'
               }
dictStateFips = {'FIPS':'fips','Name':'shrtCounty'}
dictStateCensus = {'State':'state','Code':'stateCode','Short':'abrState'}
# =============================================================================

# =============================================================================
def rename_variables (df,dictRename) :
    #   Format : dictRename{'start':'end'}
    for keyVar in dictRename.keys() :
        if keyVar in df.columns:
            df.rename(columns={keyVar:dictRename[keyVar]},inplace=True)
        ####
    ####
    return df
####
def isolate_better (stri , start , end, b_end = 0) :
    stri        = str(stri)
    strShort    = ''
    posStart    = 0
    posEnd      = 0

    if b_end==1 :
        posEnd      = stri.find(end)
        strShort    = stri[:posEnd]
        strShort    = reverse(strShort)
        start       = reverse(start)
        posStart    = posEnd - strShort.find(start)
    #
    else :
        posStart    = stri.find(start)+len(start)
        strShort    = stri[posStart:]
        posEnd      = posStart + strShort.find(end)
        if posEnd > -1 :  strEnd = stri[posStart:posEnd]
        else : strEnd = stri[posStart:]
    #
    return strEnd
####
def merge_track (left,right,by,how='outer') :
    left['fromSource'] = 1
    right['fromMerger'] = 1
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
def cap_city (x) :
    if x.find(' city') > -1 : x = x.replace(' city',' City')
    return x
####
# =============================================================================






# =============================================================================
if __name__ == '__main__' :
    print(__doc__)
    
    
    
    
    
    
    
    
    
    
    #   Import FIPS/County/State lists
    dfStateFips = read_csv('data/census FIPS Codes.txt',delimiter='\t')
    
    
    
    
    
    
    
    
    
    
    #   Check for Duplicated FIPS
    #   12 dupes found--from renamed counties
    #   Hoonah-Angoon / Skagway, AK     ; Wrangell / Petersburg, AK ;
    #   Manua / Ofu / Olosega, AS       ; Aguijan / Tinian, MP ;
    #   Cocos Island, MP / Guam, GU
    print(len(dfStateFips[dfStateFips['FIPS'].duplicated()]))
    print(dfStateFips[dfStateFips['FIPS'].duplicated(keep=False)])
    dfStateFips.rename(columns={'FIPS':'fips'},inplace=True)
    listFipFix = [(2280,'Wrangell-Petersburg','AK'),
                  (2232,'Skagway-Hoonah-Angoon','AK'),
                  (60020,'Manua','AS'),
                  (66010,'Guam','GU'),
                  (69120,'Tinian','MP')]
    for fipFix in listFipFix :
        dfStateFips = dfStateFips[dfStateFips['fips']!=fipFix[0]]
        dictNewRow = {'fips':fipFix[0],'Name':fipFix[1],'State':fipFix[2]}
        dfStateFips = dfStateFips.append(dictNewRow,ignore_index=True)
    ####
    print(len(dfStateFips[dfStateFips['fips'].duplicated()]))
    print(dfStateFips[dfStateFips['fips'].duplicated(keep=False)])
    
    
    
    
    
    
    
    
    
    
    ####    Get ZIP-County-FIPS, Rename, Merge
    dfZipFips = read_csv('data/ZIP-COUNTY-FIPS_2018-03.csv')
    dfZipFips.rename(columns={'STCOUNTYFP':'fips'},inplace=True)
    dfMerge = merge_track(dfZipFips,dfStateFips,'fips','left')[0]
    
    print('\nFrom Zip list source:')
    print(dfMerge[['fips','ZIP','COUNTYNAME','STATE']][(dfMerge['MergeSuccess']==0)&(dfMerge['fromSource']==1)])
    print('\nFrom Fip list merger:')
    print(dfMerge[['fips','Name','State']][(dfMerge['MergeSuccess']==0)&(dfMerge['fromMerger']==1)])
    print('Number of blank entries so far '+str(dfMerge.isna().sum().sum()))
    
    dfMerge = dfMerge[['fips','ZIP','CITY','STATE','COUNTYNAME','CLASSFP','Name']]
    
    
    
    
    
    
    
    
    
    
    ####    Import State/State-code
    dfStateCensus = read_csv('data/census State Codes.txt',delimiter='\t')
    dfMerge.rename(columns={'STATE':'Short'},inplace=True)
    dfMerge = merge_track(dfMerge,dfStateCensus,'Short','left')[0]
    dfMerge = dfMerge.drop(columns=['fromSource','fromMerger','MergeSuccess'])
    
    dictRename = {'ZIP':'zip',
                  'CITY':'city',
                  'Short':'abrState',
                  'COUNTYNAME':'county',
                  'CLASSFP':'fipsCode',
                  'Name':'abrCounty',
                  'State':'state',
                  'Code':'stateCode'}
    dfMerge = rename_variables(dfMerge,dictRename)
    
    
    
    
    
    
    
    
    
    
    ####    Check if FIPS overlap Counties / States
    listFIPDupe = list(dfMerge['fips'][dfMerge['fips'].duplicated(keep=False)].unique())
    for fips in listFIPDupe :
        listCounty = list(dfMerge['county'][dfMerge['fips']==fips].unique())
        listState = list(dfMerge['abrState'][dfMerge['fips']==fips].unique())
        intDupe = len(listCounty)
        if intDupe!=1 : print('Fip '+str(fips)+' is in '+str(listState)+' '+str(listCounty))
    #   endfor
    
    
    
    
    
    
    
    
    
    
    ####    Choose to export data
    inptExport = input('Export Data (Y/N):')
    if inptExport=='Y' : dfMerge.to_csv('data/censusMaster.csv',index=None)
    
#   endif
# =============================================================================



























