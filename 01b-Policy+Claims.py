# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:14:43 2020
@author: RC

Input files:        data/source/FimaNfipClaims.csv,
                    data/source/FimaNfipPolicies.csv,
Preceeding Script:  00-Fip Master.py
Output files:       data/working/policies.csv,
                    data/working/claims.csv,
                    data/working/policy+claims.csv
Following Script:   02-takeup-rate

Require multiprocessing package, must be run in console window, not 
psuedo-console

This takes csvs from the pre-2020 NFIP data and creates the policy+claims data.
This is the active version of this file. Note that policy data is in 10 
separate csvs. Having all ten active may cause crashes. This code takes steps
to minimize the likelihood of crashes.
Note that policy data is list of policies with start and end dates, must run 
through months in date-range to take stats of active policies in each month.

"""






# =============================================================================
from pandas import *
from math import *
from multiprocessing import *
from ast import *
import datetime
import time
from os import listdir
from os.path import isfile, join
#set_option('display.max_rows',100)
options.mode.chained_assignment = None
intStartYear = 2010          # Year to start collecting data, 1st January, inclusive
intStopYear = 2020          # Year to stop collecting data, 31st December, inclusive
listGroup = ['year','month','fips','binSFHA']
binDebug = 0
intDebug = 'AL'
# =============================================================================






# =============================================================================
# Define Dictionaries for Keep/Drop Variables ; Flood Zone Inclusion
# =============================================================================

#   Column names after rename_variables function
dictPolicies = {'agricultureStructureIndicator'     :0,                        # Need to remove ==1 row
              'baseFloodElevation'                  :0,
              'basementEnclosureCrawlspace'         :0,
              'censusTract'                         :'concat',
              'cancellationDateOfFloodPolicy'       :0,
              'condominiumIndicator'                :0,
              'construction'                        :0,
              'fips'                                :1,
              'crsClassCode'                        :0,                        # Not sure what this is
              'deductibleAmountInBuildingCoverage'  :'sum',
              'deductibleAmountInContentsCoverage'  :'sum',
              'elevatedBuildingIndicator'           :0,
              'elevationCertificateIndicator'       :0,
              'elevationDifference'                 :0,
              'federalPolicyFee'                    :0,
              'floodZone'                           :'concat',
              'hfiaaSurcharge'                      :0,
              'houseWorship'                        :0,
              'latitude'                            :'mean',
              'longitude'                           :'mean',
              'locationOfContents'                  :0,
              'lowestAdjacentGrade'                 :0,
              'lowestFloorElevation'                :0,
              'nonProfitIndicator'                  :0,
              'numberOfFloorsInTheInsuredBuilding'  :0,
              'obstructionType'                     :0,
              'occupancyType'                       :1,                        # Keep 1,2,3 ; drop 4, 6
              'originalConstructionDate'            :0,                        # Convert to building age, average
              'originalNBDate'                      :0,
              'policyCost'                          :'sum',
              'policyCount'                         :'sum',
              'policyEffectiveDate'                 :1,                        # Convert to coded start month+year
              'policyTerminationDate'               :1,                        # Convert to coded end month+year
              'policyTermIndicator'                 :0,
              'postFIRMConstructionIndicator'       :0,
              'primaryResidenceIndicator'           :0,
              'state'                               :0,
              'zip'                                 :'concat',
              'rateMethod'                          :0,
              'regularEmergencyProgramIndicator'    :0,
              'reportedCity'                        :'concat',
              'smallBusinessIndicatorBuilding'      :0,
              'totalBuildingInsuranceCoverage'      :'sum',
              'totalContentsInsuranceCoverage'      :'sum',
              'totalInsurancePremiumOfThePolicy'    :'sum',
              'id'                                  :0
              }


#   Column names after rename_variable function
dictClaims = {'agricultureStructureIndicator'               :0,
              'amountPaidOnBuildingClaim'                   :'sum',
              'amountPaidOnContentsClaim'                   :'sum',
              'amountPaidOnIncreasedCostOfComplianceClaim'  :'sum',
              'asOfDate'                                    :0,
              'baseFloodElevation'                          :0,
              'basementEnclosureCrawlspace'                 :0,
              'censusTract'                                 :'concat',
              'communityRatingSystemDiscount'               :0,
              'condominiumIndicator'                        :0,
              'fips'                                        :1,
              'dateOfLoss'                                  :1,
              'elevatedBuildingIndicator'                   :0,
              'elevationCertificateIndicator'               :0,
              'elevationDifference'                         :0,
              'floodZone'                                   :'concat',
              'houseWorship'                                :0,
              'latitude'                                    :'mean',
              'locationOfContents'                          :0,
              'longitude'                                   :'mean',
              'lowestAdjacentGrade'                         :0,
              'lowestFloorElevation'                        :0,
              'nonProfitIndicator'                          :0,
              'numberOfFloorsInTheInsuredBuilding'          :0,
              'obstructionType'                             :0,
              'occupancyType'                               :1,                # Keep 1, 2, 3, drop 4, 6
              'originalConstructionDate'                    :0,                # Convert to building age
              'originalNBDate'                              :0,
              'policyCount'                                 :'sum',
              'postFIRMConstructionIndicator'               :0,
              'primaryResidence'                            :0,
              'rateMethod'                                  :0,
              'reportedCity'                                :'concat',
              'zip'                                         :'concat',
              'smallBusinessIndicatorBuilding'              :0,
              'state'                                       :0,
              'totalBuildingInsuranceCoverage'              :'sum',
              'totalContentsInsuranceCoverage'              :'sum',
              'year'                                        :1,
              'id'                                          :0
              }
            

#   ==1 code as binSFHA==1 ; ==0 code binSFHA==0
dictFloodZone = {'A'    :1,
                 'A01'  :1,
                 'A02'  :1,
                 'A03'  :1,
                 'A04'  :1,
                 'A05'  :1,
                 'A06'  :1,
                 'A07'  :1,
                 'A08'  :1,
                 'A09'  :1,
                 'A10'  :1,
                 'A11'  :1,
                 'A12'  :1,
                 'A13'  :1,
                 'A14'  :1,
                 'A15'  :1,
                 'A16'  :1,
                 'A17'  :1,
                 'A18'  :1,
                 'A19'  :1,
                 'A20'  :1,
                 'A21'  :1,
                 'A22'  :1,
                 'A23'  :1,
                 'A24'  :1,
                 'A25'  :1,
                 'A26'  :1,
                 'A27'  :1,
                 'A28'  :1,
                 'A29'  :1,
                 'A30'  :1,
                 'A99'  :1,
                 'AA'   :1,
                 'AE'   :1,
                 'AH'   :1,
                 'AHB'  :1,
                 'AO'   :1,
                 'AOB'  :1,
                 'AR'   :1,
                 'AS'   :1,
                 'B'    :0,
                 'C'    :0,
                 'D'    :0,
                 'V'    :1,
                 'V01'  :1,
                 'V02'  :1,
                 'V03'  :1,
                 'V04'  :1,
                 'V05'  :1,
                 'V06'  :1,
                 'V07'  :1,
                 'V08'  :1,
                 'V09'  :1,
                 'V10'  :1,
                 'V11'  :1,
                 'V12'  :1,
                 'V13'  :1,
                 'V14'  :1,
                 'V15'  :1,
                 'V16'  :1,
                 'V17'  :1,
                 'V18'  :1,
                 'V19'  :1,
                 'V20'  :1,
                 'V21'  :1,
                 'V22'  :1,
                 'V23'  :1,
                 'V24'  :1,
                 'V27'  :1,
                 'V30'  :1,
                 'VE'   :1,
                 'X'    :0,
                 ###
                 'A0B'  :1,
                 'A00'  :1,
                 'ALT'  :0,
                 '*'    :0,
                 'V8'   :1,
                 'A0'   :1,
                 'ARE'  :0,
                 'V9'   :1,
                 'EMG'  :0,
                 'X 0'  :0,
                 'A E'  :1,
                 'E'    :0,
                 '00X'  :0
                 }
# =============================================================================






# =============================================================================
# Define Functions
# =============================================================================
##  Takes str and subsets it by start & stop substrings, exclusively. 
#   Originally dev'ed for use with html; used here for expediency.
def reverse (stri) :
    x = ""
    for i in stri :
        x = i + x
    return x
#
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
    #
    return stri[posStart:posEnd]
#
##
def programTimer (timeStart) :
    timeEnd = time.time()           # Get current time for end
    timeDiff = timeEnd - timeStart  # Difference = number of seconds

    minutes = int(timeDiff//60)
    seconds = round(timeDiff - 60*minutes,3)

    hours =  int(minutes//60)
    minutes = minutes - 60*hours

    days = int(hours//24)
    hours = hours - 24*days

    if days!=0 :
        return (str(days)+' Days, '+str(hours)+' Hours, '+str(minutes)
              +' Minutes, and '+str(seconds)+' Seconds!')
    elif hours!=0 :
        return (str(hours)+' Hours, '+str(minutes)
              +' Minutes, and '+str(seconds)+' Seconds!')
    elif minutes!=0 :
        return (str(minutes)+' Minutes, and '+str(seconds)+' Seconds!')
    else :
        return (str(seconds)+' Seconds!')
#
#
##  Merges two DFs by By variable list, defaults to 'outer'. Tracks source of
#   each row, reports on number of successful matches.
def merge_track (left,right,by,how='outer') :
    #   Fixed version!
    left['fromSource'] = 1
    right['fromMerger'] = 1
    
    if (how=='left')&(len(right[right[by].duplicated()])>0) :
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
#
##  Takes DF and renames columns if those columns appear 
def rename_variables (df,dictRename) :
    for keyVar in dictRename.keys() :
        if keyVar in df.columns:
            df.rename(columns={keyVar:dictRename[keyVar]},inplace=True)
        #
    #
    return df
#
##  Takes DF, str, list of column names in DF. Attached str to column names 
#   not list in lstSkip
def add_variable_prefix (df,strAdd,lstSkip) :
    lstVariables = list(df)
    lstVariables = [x for x in lstVariables if x not in lstSkip]
    for var in lstVariables :
        df = df.rename(columns={var:strAdd+var[:1].capitalize()+var[1:]})
    return df
#
##
def drop_ag_church (df) :
    intRowCount = len(df)
    
    df = df[(df['occupancyType']==1)|
            (df['occupancyType']==2)|
            (df['occupancyType']==3)]
    
    df = df[(df['agricultureStructureIndicator']!='Y')
            &(df['houseWorship']!='Y')]
    
    intRowChange = intRowCount - len(df)
    print('Rows dropped: '+str(intRowChange))
    return df
#
##
def drop_year (df,intStartYear,intStopYear) :
    intRowCount = len(df)
    
    if 'year' in df.columns : 
        df = df[(df['year']<=intStopYear)&(df['year']>=intStartYear)]
    
    intRowChange = intRowCount - len(df)
    print('Rows dropped: '+str(intRowChange))
    return df
#
##
def remove_unused_vars (df,dictVars) :
    #   Get drop list
    intColumns = len(list(df))
    lstDropVars = [var for var,value in dictVars.items() if value!=0]
    
    #   Drop vars
    df = df[lstDropVars]
    print('Number of columns dropped: '+str(intColumns-len(list(df))))
    return df
#
##
# def add_month_from_claim (df) :
#     df['month'] = ''
#     for index, row in df.iterrows() :
#         df.at[index,'monthOfLoss'] = isolate_better(row['dateOfLoss'],'-','-')
#     #
#     return df
# #
def extract_year_month (df,col) :
    # df['year'] = 0
    # df['month'] = 0
    # for index, row in df.iterrows() :
    #     df.at[index,'year'] = isolate_better(row[col],'','-')
    #     df.at[index,'month'] = isolate_better(row[col],'-','-')
    # #
    df['year'] = df[col].apply(isolate_better,args=('','-'))
    df['year'] = to_numeric(df['year'],errors='coerce')
    df['month'] = df[col].apply(isolate_better,args=('-','-'))
    df['month'] = to_numeric(df['month'],errors='coerce')
    
    return df['year'], df['month']
#
##
def bin_flood_zone (df,dictFloodZone) :
    #df['binSFHA'] = 0
    lstFloodZone = [x for x,value in dictFloodZone.items() if value==1]
    listFZ2 = [lstFloodZone]
    
    def apply_bin_fz (x,lstFloodZone) :
        #print(lstFloodZone)
        binSFHA = 0
        if x in lstFloodZone : binSFHA = 1
        return binSFHA
    #
    
    df['binSFHA'] = df['floodZone'].apply(apply_bin_fz,args=(listFZ2))
    # for index,row in df.iterrows() :
    #     if row['floodZone'] in lstFloodZone : df.at[index,'binSFHA'] = 1
    print('Percent of rows not in flood zones: '
          +str(round((len(df[df['binSFHA']==0])/len(df))*100,2))+'%')
    #
    lstA = ['A','A01','A02','A03','A04','A05','A06','A07','A08','A09','A10',
            'A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21',
            'A22','A23','A24','A25','A26','A27','A28','A29','A30','A99','AA',
            'AE','AH','AHB','AO','AOB','AR','AS','A0B','A00',
            'A0','A E']
    lstV = ['V','V01','V02','V03','V04','V05','V06','V07','V08','V09','V10',
            'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21',
            'V22','V23','V24','V27','V29','V30','VE',
            'V8','V9']
    df['floodZone'] = df['floodZone'].replace(lstA,'A')
    df['floodZone'] = df['floodZone'].replace(lstV,'V')
    print('Frequency of Flood Zone Code:\n'
          +str(df['floodZone'].value_counts()))
    #
    return df
#
##
def encode_month (df,month,year,intStartYear) :
    # df['monthCode'] = 0
    df['monthCode'] = (12*(df[year]-intStartYear))+df[month]
    return df['monthCode']
#
##
def month_modulo (x) :
    m = ((x-1)%12)+1
    return m
#
##
def fix_data_type (df,lstVar) :
    for var in lstVar :
        df[var] = to_numeric(df[var],errors='coerce')
    return df
#
##
def order_cols (df,lstFirst) :
    lstColumns = list(df)
    lstColumns = [x for x in lstColumns if x not in lstFirst]
    return df[lstFirst+lstColumns]
#
##
#
##
def empty_zips (df) :
    df.dropna(subset=['zip'],inplace=True)
    return df
####
#   Combine non-numeric columns into single entries
def concat_row_collapse (dfNew) :
    df = DataFrame()
    zipp = dfNew['fips'].unique()[0]
    lstYear = list(dfNew['year'].unique())
    for year in lstYear :
        #print('Starting year '+str(year))
        lstMonth = list(dfNew[(dfNew['year']==year)
                                 ]['month'].unique())
        #
        for month in lstMonth :
            #print('Starting month '+str(month))
            lstZip = list(dfNew[(dfNew['year']==year)
                                   &(dfNew['month']==month)
                                   ]['fips'].unique())
            #
            for zipp in lstZip :
                lstBin = list(dfNew[(dfNew['year']==year)
                                       &(dfNew['month']==month)
                                       &(dfNew['fips']==zipp)
                                       ]['binSFHA'].unique())
                #
                for binn in lstBin : 
                    dictNewRow = {}
                    dictNewRow['year']          = year
                    dictNewRow['month']         = month
                    dictNewRow['fips']           = zipp
                    dictNewRow['binSFHA']       = binn
                    #
                    dfSubset = dfNew[(dfNew['year']==year)
                                        &(dfNew['month']==month)
                                        &(dfNew['fips']==zipp)
                                        &(dfNew['binSFHA']==binn)]
                    #
                    lstConcat = ['censusTract','zip','floodZone','reportedCity',
                                 'county','abrState']
                    for var in lstConcat :
                        dictNewRow[var] = list(dfSubset[var].unique())
                    #
                    df = df.append(dictNewRow, ignore_index=True)
                print(str(zipp)+' '+str(month)+' '+str(year))
                #
            #
        #
    #
    return df
####
#   Takes df with unique fip ; for each month in data range take stats for all
#   active policies in the month.
def save_policy_data (df) :
    dfCollapse = DataFrame()
    zipp = int(df['fips'].unique()[0])
    intStart = df['startCode'].min()
    intEnd = df['endCode'].max()
    
    listConcat = ['censusTract','zip','floodZone','reportedCity','county',
                  'abrState']
    listSum = ['deductibleAmountInBuildingCoverage','policyCost',
               'deductibleAmountInContentsCoverage','policyCount',
               'totalBuildingInsuranceCoverage',
               'totalContentsInsuranceCoverage',
               'totalInsurancePremiumOfThePolicy']
    listMean = ['latitude','longitude']
            
    intMonth = intStart
    while intMonth <= intEnd :
        dfSubset = df[(df['startCode']<=intMonth)&(df['endCode']>=intMonth)]
        
        listBin = list(dfSubset['binSFHA'].unique())
        for binn in listBin :
            dfSubset2 = dfSubset[dfSubset['binSFHA']==binn]
            dictNewRow = {'fips':zipp,'binSFHA':binn}
            
            dictNewRow['month'] = month_modulo(intMonth)
            dictNewRow['year']  = 2010 + floor((intMonth-1)/12)
            
            for var in listConcat :
                dictNewRow[var] = list(dfSubset2[var].unique())
            for var in listSum :
                dictNewRow[var] = dfSubset2[var].sum()
            for var in listMean :
                dictNewRow[var] = dfSubset2[var].mean()
                
            dfCollapse = dfCollapse.append(dictNewRow,ignore_index=True)            
        intMonth+=1
    print('finished fip: '+str(zipp))
    return dfCollapse
####
def combine_list_col (x,binNumeric=0) :
    #   Requires ast package
    
    item1 = x[0]
    item2 = x[1]
    #print(str(item1)+' '+str(item2))
    
    if (notnull(item1)) : 
        #print(item1)
        item1 = item1.replace(', nan','')
        item1 = item1.replace('nan, ','')
        item1 = literal_eval(item1)
    else : item1 = []    
    
    if (notnull(item2)) : 
        #print(item2)
        item2 = item2.replace(', nan','')
        item2 = item2.replace('nan, ','')
        item2 = literal_eval(item2)
    else : item2 = []
    
    listCombo = item1 + item2    
    listCombo = list(set(listCombo))
    listCombo = [x for x in listCombo if str(x)!='nan']
    
    if (binNumeric==1) :
        try :
            listCombo = list(map(float,listCombo))
            listCombo = list(map(int,listCombo))
        except :
            listCombo = listCombo
        #   end try
    #   end if
    
    return listCombo
####
def col_mean(x) :
    fltMean = x[~isna(x)].mean()
    return fltMean
####
def fix_fips (df,strData) :
    if strData=='Claims' :
        #   Fix broken fips ('fips',fip-start,'reportedCity',city-start,fip-new)
        listReplace = [('fips',22,    'reportedCity','MARKSVILLE',  22009),
                       ('fips',29,    'reportedCity','DONIPHAN',    29181),
                       ('fips',34,    'reportedCity','HACKENSACK',  34003),
                       ('fips',34,    'reportedCity','TOMS RIVER TOWNSHIP',  34029),
                       ('fips',34,    'reportedCity','HARVEY CEDARS',  34029),
                       ('fips',35,    'reportedCity','SAN ANTONIO',  35053),
                       ('fips',36,    'reportedCity','FIRE ISLAND PIN',  36103),
                       ('fips',36,    'reportedCity','OCEAN BAY PARK',  36103),
                       ('fips',37,    'reportedCity','OCRACOKE',  37095),
                       ('fips',37,    'reportedCity','NAGS HEAD',  37055),
                       ('fips',40,    'reportedCity','GUTHRIE',  40083),
                       ('fips',48,    'reportedCity','BEAUMONT',  48245),
                       ('fips',48,    'reportedCity','EDCOUCH',  48215),
                       ('fips',48,    'reportedCity','DICKINSON',  48167),
                       ('fips',72,    'reportedCity','SAN JUAN',  72127),
                       ('fips',72,    'reportedCity','HATO REY',  72127),
                       ('fips',78,    'reportedCity','SAINT THOMAS',  78030),
                       ('fips',22201, 'reportedCity','KENNER',  22051),
                       ('fips',51515, 'reportedCity','BEDFORD',  51019),
                        ]
        for replace in listReplace :
            df.loc[(df[replace[0]]==replace[1])
                         &(df[replace[2]]==replace[3])
                         ,replace[0]] = replace[4]
        ####
    ####
    if strData=='Policy' :
        #   Fix broken fips
        listReplace = [('fips',12,12097),
                       ('fips',37,37055),
                       ('fips',22201,22051),
                       ('fips',16,16001)]
        for replace in listReplace :
            df.loc[df[replace[0]]==replace[1],replace[0]] = replace[2]
        ####
    ####
    return df
####
def add_census_data (df) :
    dfCFip = df[df['fips'].notnull()]
    dfCZip = df[(df['fips'].isna())&(df['zip'].notnull())]
    del dfCZip['fips']

    dfCensus = read_csv('data/censusMaster.csv')
    dfFIPS = dfCensus.drop_duplicates(subset=['fips'])
    dfFIPS = dfFIPS[['fips','county','abrState','stateCode']]
    dfZIPS = dfCensus.drop_duplicates(subset=['zip'])
    dfZIPS = dfZIPS[['zip','fips','county','abrState','stateCode','crossCounty']]
    
    dfCFip = merge_track(dfCFip,dfFIPS,'fips','left')[0]
    dfCZip = merge_track(dfCZip,dfZIPS,'zip','left')[0]
    
    df = concat([dfCFip,dfCZip])
    df = df[df['MergeSuccess']==1]
    for var in ['MergeSuccess','fromMerger','fromSource','crossCounty','stateCode'] :
        del df[var]
    ####
    return df
####
def export_data (df,name,pathData):
    listFiles = [f for f in listdir(pathData) if isfile(join(pathData,f))]    
    if name+'.csv' not in listFiles : df.to_csv(pathData+name+'.csv',index=None)
    else : df.to_csv(pathData+name+'.csv',index=None,header=False,mode='a')
####
# =============================================================================










# =============================================================================
#   Main() functions
# =============================================================================
def generate_claims_data (intStartYear,
                          intStopYear,
                          dictFloodZone,
                          dictClaims,
                          listGroup,
                          binDebug,
                          intDebug
                          ) :
    ####    Import data
    print('Importing Claims data now')
    dfClaims = read_csv('data/source/FEMA2019/openFEMA_claims20190731.csv')
    
    
    
    
    
    
    
    
    
    
    ####    Rename variables to match standards
    dictNameClaim = {'agriculturestructureindicator':'agricultureStructureIndicator',
                     'amountpaidonbuildingclaim':'amountPaidOnBuildingClaim',
                     'amountpaidoncontentsclaim':'amountPaidOnContentsClaim',
                     'amountpaidonincreasedcostofcomplianceclaim':'amountPaidOnIncreasedCostOfComplianceClaim',
                     'censustract':'censusTract',
                     'countycode':'fips',
                     'dateofloss':'dateOfLoss',
                     'floodzone':'floodZone',
                     'houseworship':'houseWorship',
                     'nonprofitindicator':'nonProfitIndicator',
                     'occupancytype':'occupancyType',
                     'policycount':'policyCount',
                     'reportedcity':'reportedCity',
                     'reportedzipcode':'zip',
                     'totalbuildinginsurancecoverage':'totalBuildingInsuranceCoverage',
                     'totalcontentsinsurancecoverage':'totalContentsInsuranceCoverage',
                     'yearofloss':'year'
                     }
    dfClaims = rename_variables(dfClaims,dictNameClaim)
    
    
    
    
    
    
    
    
    
    
    ####    Fix Cols
    #   Create debug subset
    if binDebug==1 : dfClaims = dfClaims[dfClaims['state']==intDebug]
    
    #   Drop House-of-Worship, Agg, Out-of-Time-Range rows
    dfClaims = drop_ag_church(dfClaims)
    dfClaims = drop_year(dfClaims,intStartYear,intStopYear)
    
    #   Drop Unused Columns
    dfClaims = remove_unused_vars(dfClaims,dictClaims)
    
    #   Add in census data
    dfClaims = add_census_data(dfClaims)
    
    #   Gen 'month' and 'binSFHA' variables
    dfClaims['month'] = extract_year_month(dfClaims,'dateOfLoss')[1]
    dfClaims = bin_flood_zone(dfClaims,dictFloodZone)
    
    
    
    
    
    
    
    
    
    
    ####    Get lists
    lstConcat   = [x for x,value in dictClaims.items() if value=='concat']
    lstAverage  = [x for x,value in dictClaims.items() if value=='mean']
    lstSum      = [x for x,value in dictClaims.items() if value=='sum']
    
    #   Add Census Vars
    lstConcat = lstConcat+['county','abrState']
    
    #   Fix date type problems
    dfClaims = fix_data_type(dfClaims,lstAverage+lstSum)
    
    
    
    
    
    
    
    
    
    
    ####    Collapse to fips for quant cols
    timeGroupBy = time.time()
    #   Get unique key
    dfNew2 = dfClaims[listGroup].drop_duplicates()
    dfNew2 = dfNew2[listGroup]
    
    #   Sum
    df = DataFrame()
    df = dfClaims.groupby(listGroup)[lstSum].sum().reset_index()
    dfNew2 = merge_track(dfNew2,df,by=listGroup,how='left')[0]
    dfNew2 = dfNew2.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    
    #   Average
    df = DataFrame()
    df = dfClaims.groupby(listGroup)[lstAverage].mean().reset_index()
    dfNew2 = merge_track(dfNew2,df,by=listGroup,how='left')[0]
    dfNew2 = dfNew2.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    
    
    
    
    
    
    
    
    
    
    ####    Collapse to fips for str cols
    # #   Multithread
    dictClaimsData = {}
    listZip = list(dfClaims['fips'].unique())
    for zipp in listZip:
        dictClaimsData[zipp] = dfClaims[listGroup+lstConcat][dfClaims['fips']==zipp]
    #

    pool = Pool(cpu_count())
    result = pool.map(concat_row_collapse,dictClaimsData.values())
    print('done.')
    
    dfFinal = concat(result)
    
    
    
    
    
    
    
    
    
    
    ####    Combine results
    dfNew2 = merge_track(dfNew2,dfFinal,by=listGroup,how='left')[0]
    dfNew2 = dfNew2.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    
    
    
    
    
    
    
    
    
    
    ####    Add prefix and export
    dfNew2 = add_variable_prefix(dfNew2,'claims',listGroup)
    dfNew2.to_csv('data/working/claims2.csv',index=None)
####










def generate_policies_data (intStartYear,
                            intStopYear,
                            dictFloodZone,
                            dictPolicies,
                            listGroup,
                            binDebug,
                            intDebug
                            ) :
    ####    Import Data
    print('Importing Policies Data Now')
    pathFile = 'data/source/FEMA2019/openFEMA_policies20190731_'        # file path
    listFile = ['01','02','03','04','05','06','07','08','09','10']      # file names
    
    #   Read through files -- import / make fixes / export to single csv
    listHeader = []
    dfPolicies = DataFrame()
    for file in listFile :
        print('Starting file : '+str(file))
        if file=='01' :
            #   import data
            dfPolicies = read_csv(pathFile+file+'.csv')
            
            #   Rename variables to match standard
            dictNamePolicy = {'agriculturestructureindicator':'agricultureStructureIndicator',
                             'censustract':'censusTract',
                             'countycode':'fips',
                             'deductibleamountinbuildingcoverage':'deductibleAmountInBuildingCoverage',
                             'deductibleamountincontentscoverage':'deductibleAmountInContentsCoverage',
                             'floodzone':'floodZone',
                             'houseofworshipindicator':'houseWorship',
                             'occupancytype':'occupancyType',
                             'policycost':'policyCost',
                             'policycount':'policyCount',
                             'policyeffectivedate':'policyEffectiveDate',
                             'policyterminationdate':'policyTerminationDate',
                             'reportedcity':'reportedCity',
                             'reportedzipcode':'zip',
                             'totalbuildinginsurancecoverage':'totalBuildingInsuranceCoverage',
                             'totalcontentsinsurancecoverage':'totalContentsInsuranceCoverage',
                             'totalinsurancepremiumofthepolicy':'totalInsurancePremiumOfThePolicy'
                             }
            dfPolicies = rename_variables(dfPolicies,dictNamePolicy)
            listHeader = list(dfPolicies)
        else : dfPolicies = read_csv(pathFile+file+'.csv',names=listHeader)
        #   end if
        print('Adjusting vars')
        #   Drop House-of-Worship, Agg, Out-of-Time-Range rows
        dfPolicies = drop_ag_church(dfPolicies)
        
        #   Drop Unused Columns
        dfPolicies = remove_unused_vars(dfPolicies,dictPolicies)
        
        #   Gen 'binSFHA' variable
        dfPolicies = bin_flood_zone(dfPolicies,dictFloodZone)
        
        #   Add in census data
        #dfPolicies = fix_fips(dfPolicies,'Policy')
        dfPolicies = add_census_data(dfPolicies)
        
        # #   Do Something w Empty Zips
        # dfPolicies = empty_zips(dfPolicies)
        
        #   INTS : Gen Stop month code
        intEndCode = (intStopYear-intStartYear+1)*12
        
        #   COLUMNS : Gen start and stop time variables, encode month numbers
        dfPolicies['yearStart'],dfPolicies['monthStart'] = extract_year_month(
            dfPolicies,'policyEffectiveDate')
        dfPolicies['yearEnd'],dfPolicies['monthEnd'] = extract_year_month(
            dfPolicies,'policyTerminationDate')
        
        dfPolicies['startCode'] = encode_month(
            dfPolicies[['monthStart','yearStart']],'monthStart','yearStart',
            intStartYear)
        dfPolicies['endCode'] = encode_month(
            dfPolicies[['monthEnd','yearEnd']],'monthEnd','yearEnd',
            intStartYear)
        
        #   Grab list of states
        listState = list(dfPolicies['abrState'].unique())
        
        #   Drop remaining variables that can be dropped
        dfPolicies = dfPolicies.drop(columns=['year','month','yearEnd',
                                              'policyEffectiveDate','policyTerminationDate',
                                              'yearStart','monthStart','monthEnd'])
        dfPolicies = dfPolicies[dfPolicies['startCode']>0]
        
        #   Export data to files
        print('Exporting file '+str(file))
        #listState = list(dfPolicies['abrState'].unique())
        for state in listState :
            export_data(dfPolicies[dfPolicies['abrState']==state],
                        state,'data/working/fema2019/')
        #   endfor
    #   endfor
    del dfPolicies # for memory
    

    







    ####    Make lists
    lstConcat   = [x for x,value in dictPolicies.items() if value=='concat']
    lstAverage  = [x for x,value in dictPolicies.items() if value=='mean']
    lstSum      = [x for x,value in dictPolicies.items() if value=='sum']
    
    #   Add Census Vars
    lstConcat = lstConcat+['county','abrState']
    
    
    
    
    
    
    
    
    
    
    ####    Read through created csvs / collapse into fips+month
    pathData = 'data/working/fema2019/'
    listFiles = [f for f in listdir(pathData) if isfile(join(pathData,f))]   
    dfOutput = DataFrame()
    
    for file in listFiles :
        print('Start : '+str(file))
        dfPolicies = read_csv(pathData+file)
        
        #   Fix date type problems
        dfPolicies = fix_data_type(dfPolicies,lstAverage+lstSum)
        
        #   Create list of dfs for each fip    
        dictPolicyData = {}
        for zipp in list(dfPolicies['fips'].unique()) :
            dictPolicyData[zipp] = dfPolicies[dfPolicies['fips']==zipp]
    
        #   Run save_policy_data on each df ; return list of new dfs ; combine
        #   back into a single df
        pool = Pool(cpu_count())
        result = pool.map(save_policy_data,dictPolicyData.values())
        timeConcat = time.time()
        dfNew2 = concat(result)
        print(programTimer(timeConcat))
        ####
        
        #   Add source prefix, export data
        dfNew = add_variable_prefix(dfNew2,'policy',listGroup)
        
        #   Add to output
        dfOutput = dfOutput.append(dfNew,True)
    #   end for
    
    
    
    
    
    
    
    
    
    ####    Export
    dfOutput.to_csv('data/working/policies2.csv',index=None)
####










def merge_claims_policies (listGroup) :
    ####    Import data
    dfPolicies = read_csv('data/working/policies2.csv')
    dfClaims = read_csv('data/working/claims2.csv')
    
    
    
    
    
    
    
    
    
    
    ####    Merge data
    print('Running Merge Now')
    dfPC = merge_track(dfPolicies,dfClaims,by=listGroup)[0]
    
    #   Drop Merger debug variables
    dfPC = dfPC.drop(columns=['fromSource','fromMerger'])
    
    #   Fix col entries
    listVars = ['policyCensusTract','claimsCensusTract',
                'policyCounty','claimsCounty',
                'policyFloodZone','claimsFloodZone',
                'policyLatitude','claimsLatitude',
                'policyLongitude','claimsLongitude',
                'policyReportedCity','claimsReportedCity',
                'policyZip','claimsZip']
    listNan = ['[nan]','[]','nan']

    for var in listVars :
        dfPC[var] = dfPC[var].replace(listNan,nan)
    #   endfor
    
    
    
    
    
    
    
    
    
    
    #   Combine cols / Rename
    dfPC['censusTract'] = dfPC[['policyCensusTract','claimsCensusTract']].apply(combine_list_col,1,args=(1,))
    dfPC['county'] = dfPC[['policyCounty','claimsCounty']].apply(combine_list_col,1)
    dfPC['floodZone'] = dfPC[['policyFloodZone','claimsFloodZone']].apply(combine_list_col,1)
    dfPC['city'] = dfPC[['policyReportedCity','claimsReportedCity']].apply(combine_list_col,1)
    dfPC['zip'] = dfPC[['policyZip','claimsZip']].apply(combine_list_col,1,args=(1,))
    dfPC['abrState'] = dfPC[['policyAbrState','claimsAbrState']].apply(combine_list_col,1)
    
    dfPC['latitude'] = dfPC[['policyLatitude','claimsLatitude']].apply(col_mean,1)
    dfPC['longitude'] = dfPC[['policyLongitude','claimsLongitude']].apply(col_mean,1)
    
    dfPC = dfPC.drop(columns=['policyCensusTract','claimsCensusTract',
                              'policyCounty','claimsCounty',
                              'policyFloodZone','claimsFloodZone',
                              'policyLatitude','claimsLatitude',
                              'policyLongitude','claimsLongitude',
                              'policyReportedCity','claimsReportedCity',
                              'policyZip','claimsZip',
                              'policyAbrState','claimsAbrState'])
    
    dictRename = {'policyDeductibleAmountInBuildingCoverage':'policyDeductibleAmountinBuildingCoverage',
                  'policyDeductibleAmountInContentsCoverage':'policyDeductibleAmountinContentsCoverage',
                  'policyTotalInsurancePremiumOfThePolicy':'policyTotalInsurancePremiumofthePolicy',
                  'claimsAmountPaidOnBuildingClaim':'claimsAmountPaidonBuildingClaim',
                  'claimsAmountPaidOnContentsClaim':'claimsAmountPaidonContentsClaim',
                  'claimsAmountPaidOnIncreasedCostOfComplianceClaim':'claimsAmountPaidonIncreasedCostofComplianceClaim',
                  }
    dfPC = rename_variables(dfPC,dictRename)

    
    
    
    
    
    
    
    
    
    #### Order cols
    dfPC = dfPC[['year','month','fips','binSFHA','censusTract','county',
                 'floodZone','city','abrState','latitude','longitude',
                 'policyDeductibleAmountinBuildingCoverage',
                 'policyDeductibleAmountinContentsCoverage',
                 'policyPolicyCost','policyPolicyCount',
                 'policyTotalBuildingInsuranceCoverage',
                 'policyTotalContentsInsuranceCoverage',
                 'policyTotalInsurancePremiumofthePolicy',
                 'claimsAmountPaidonBuildingClaim',
                 'claimsAmountPaidonContentsClaim',
                 'claimsAmountPaidonIncreasedCostofComplianceClaim',
                 'claimsPolicyCount',
                 'claimsTotalBuildingInsuranceCoverage',
                 'claimsTotalContentsInsuranceCoverage',
                 'zip','MergeSuccess']]
    
    #   Drop extra lines--this should drop nothing, just fail safe
    dfPC = dfPC[((dfPC['policyPolicyCount']!=0)
                 &(dfPC['policyPolicyCount'].notnull()))
                |((dfPC['claimsPolicyCount']!=0)
                  &(dfPC['claimsPolicyCount'].notnull()))]
    
    #   Export merged data
    dfPC.to_csv('data/working/policy+claims2.csv',index=None)
#
# =============================================================================





 
 





# =============================================================================
# Script Body
# =============================================================================

if __name__ == '__main__' :
    #   Generate claims data
    generate_claims_data(intStartYear,intStopYear,dictFloodZone,dictClaims,
                          listGroup,binDebug,intDebug) 
    
    #   Generate policy data
    generate_policies_data(intStartYear,intStopYear,dictFloodZone,
                            dictPolicies,listGroup,binDebug,intDebug) 
    
    #   Combine and orginize policy and claims
    merge_claims_policies(listGroup) 
# #
    
# =============================================================================





# =============================================================================
#       Scrap code
# =============================================================================

# dictClaims      = {key:value for key, value in dictClaims.items() if value==1}
# dictPolicies    = {key:value for key, value in dictPolicies.items() if value==1}

# list(dictClaims.keys())
# list(dictPolicies.keys())





    #                 for var in lstConcat :
#                     dictNewRow[var] = list(dfSubset[(dfSubset['startCode']<=intMonthCounter)&
#                                                     (dfSubset['endCode']>=intMonthCounter)
#                                                     ][var].unique())
#                 #
#                 for var in lstAverage :
#                     dictNewRow[var] = dfSubset[(dfSubset['startCode']<=intMonthCounter)&
#                                                (dfSubset['endCode']>=intMonthCounter)
#                                                ][var].mean()
#                 #
#                 for var in lstSum :
#                     dictNewRow[var] = dfSubset[(dfSubset['startCode']<=intMonthCounter)&
#                                                (dfSubset['endCode']>=intMonthCounter)
#                                                ][var].sum()
#                 #
#                 dfNew = dfNew.append(dictNewRow,ignore_index=True)       
    # #   Get unique key
    # dfNew2 = dfNew[listGroup].drop_duplicates()
    # dfNew2 = dfNew2[listGroup]
    # print('New set defined')
    # #   Sum
    # df = DataFrame()
    # df = dfNew.groupby(listGroup)[lstSum].sum().reset_index()
    # dfNew2 = merge_track(dfNew2,df,by=listGroup,how='left')[0]
    # dfNew2 = dfNew2.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    # print('Sum set done')
    # #   Average
    # df = DataFrame()
    # df = dfNew.groupby(listGroup)[lstAverage].mean().reset_index()
    # dfNew2 = merge_track(dfNew2,df,by=listGroup,how='left')[0]
    # dfNew2 = dfNew2.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    # print('Mean set done')

    # # #   Multithread
    # dictPolicyData = {}
    # listZip = list(dfNew['zip'].unique())
    # for zipp in listZip:
    #     dictPolicyData[zipp] = dfNew[listGroup+lstConcat][dfNew['zip']==zipp]
    # #

    # pool = Pool(cpu_count())
    # result = pool.map(concat_row_collapse,dictPolicyData.values())
    # print('done.')
    
    # dfFinal = DataFrame()
    # for df in result :
    #     dfFinal = dfFinal.append(df,ignore_index=True)
    # #
    
    # dfNew2 = merge_track(dfNew2,dfFinal,by=listGroup,how='left')[0]
    # dfNew2 = dfNew2.drop(columns=['fromMerger','MergeSuccess','fromSource'])



    # #   Do Something with empty zips
    # dfPolicies = empty_zips(dfPolicies)
    
    #   Cycle through months in range, consolidate rows

    #   Version 2 -- Group By
    
    # intMonthCounter = 1
    # dfNew = DataFrame()
    # while intMonthCounter <= intEndCode :
    #     df = dfPolicies[(dfPolicies['startCode']<=intMonthCounter)
    #                     &(dfPolicies['endCode']>=intMonthCounter)]
    #     df['year']          = intStartYear + floor((intMonthCounter-1)/12)
    #     df['month']         = month_modulo(intMonthCounter)
    #     dfNew = dfNew.append(df,ignore_index=True)
    #     print('Month Added: '+str(intMonthCounter))
    #     intMonthCounter+=1

    # timeLoop = time.time()
    # dfNew2 = DataFrame()
    # for df in result :
    #     dfNew2 = dfNew2.append(df,ignore_index=True)
    # print(programTimer(timeLoop))
    






'''
combine_list_col

'year',
'month',
'zip',
'binSFHA',

'policiesCensustract',
'claimsCensustract',

'policiesCountycode',
'claimsCountycode',

'policiesFloodzone',
'claimsFloodzone',

'policiesLatitude',
'policiesLongitude',
'claimsLatitude',
'claimsLongitude',

'policiesReportedcity',
'claimsReportedcity',

'policiesState',
'claimsState',


'policiesDeductibleamountinbuildingcoverage',
'policiesDeductibleamountincontentscoverage',
'policiesPolicycost',
'policiesPolicycount',
'policiesTotalbuildinginsurancecoverage',
'policiesTotalcontentsinsurancecoverage',
'policiesTotalinsurancepremiumofthepolicy',
'claimsAmountpaidonbuildingclaim',
'claimsAmountpaidoncontentsclaim',
'claimsAmountpaidonincreasedcostofcomplianceclaim',
'claimsPolicycount',
'claimsTotalbuildinginsurancecoverage',
'claimsTotalcontentsinsurancecoverage',

'MergeSuccess'

'''








 

   
#     #   Add file source to variable names
#     lstVariables = list(df)
#     for var in lstVariables :
#         df = df.rename(columns={var:varname+var.capitalize()})
#     #
    
#     #   Export file changes
#     df.to_csv('data/Policy+Claims/'+filename,index=False)
# #

#dictFloodZone = {'AE':1,'X':0}
#dfClaims['floodZone'] in list(dictFloodZone.keys())



# =============================================================================
#                           ==Data To-Do==
#   Remove ag==1 & church==1 rows
#   Drop columns dict.value==0
#   
#   Create Zip+FloodZone
#       binFlood = 1 if floodZone in { }
#       binFlood = 0 if floodZone in { }
#   
#   Grouping Vars: Year + Month + Zip + binSFHA
#
#   Average Variables w/in Group
#
#
#   Aggregate Variables w/in Group
#
#
#   Concat Variables w/in Group
#
#
#
#
# =============================================================================


# =============================================================================
#           == For Claims==
# Grouping vars: 'yearOfLoss','monthOfLoss','state','reportedZipcode','binSFHA'
#                
#
# Average vars (use 'policyCount' to weight row?): 
#               'latitude','longitude'
# 'amountPaidOnBuildingClaim',
# 'amountPaidOnContentsClaim',
# 'amountPaidOnIncreasedCostOfComplianceClaim',
# 'totalBuildingInsuranceCoverage',
# 'totalContentsInsuranceCoverage',
#
# Concat vars: 'reportedCity','countyCode','censusTract'
#
# Drop vars: 'dateOfLoss','floodZone'
#
# ?:  'elevatedBuildingIndicator',
#     'lowestAdjacentGrade',
#     'occupancyType',
#     'originalConstructionDate',
#     'smallBusinessIndicatorBuilding',
#
# =============================================================================

# =============================================================================
#           == For Policies==
# Grouping vars: 
#                
#
# Average vars (use 'policyCount' to weight row?): 
#               'latitude','longitude'
#
#
#
#
#
#
# Concat vars: 'reportedCity','countyCode','censusTract'
#
# Drop vars: 'dateOfLoss','floodZone'
#
# ?:  'elevatedBuildingIndicator',
#     'lowestAdjacentGrade',
#     'occupancyType',
#     'originalConstructionDate',
#     'smallBusinessIndicatorBuilding',
#
# =============================================================================
# """
#  'censusTract',
#  'countyCode',
#  'crsClassCode',
#  'deductibleAmountInBuildingCoverage',
#  'deductibleAmountInContentsCoverage',
#  'elevationBuildingIndicator',
#  'floodZone',
#  'latitude',
#  'longitude',
#  'lowestAdjacentGrade',
#  'occupancyType',
#  'originalConstructionDate',
#  'policyCost',
#  'policyCount',
#  'policyEffectiveDate',
#  'policyTerminationDate',
#  'propertyState',
#  'reportedZipCode',
#  'reportedCity',
#  'smallBusinessIndicatorBuilding',
#  'totalBuildingInsuranceCoverage',
#  'totalContentsInsuranceCoverage',
#  'totalInsurancePremiumOfThePolicy'
# """

# {'crsClassCode',
#  'deductibleAmountInBuildingCoverage',
#  'deductibleAmountInContentsCoverage',
#  'policyCost',
#  'policyEffectiveDate',
#  'policyTerminationDate',
#  'totalInsurancePremiumOfThePolicy'}




# =============================================================================
# #   Find unique variables
# setClaims = set(list(dictClaims.keys()))
# setPolicies = set(list(dictPolicies.keys()))
# setClaims - setPolicies
# setPolicies - setClaims
# setPolicies | setClaims
# =============================================================================

# =============================================================================
#           From Policy disaster
#     #   Version 1
#     timeLoop = time.time()
#     dfNew = DataFrame(columns=list(dfPolicies))
#     dfNew.to_csv('data/working/policies.csv',index=None)
#     print('Variable Tranformation Complete ; Beginning Row Consolidation Now')
#     lstZip = list(dfPolicies['zip'].unique())
#     intIterTotal = len(lstZip)
#     intIterCount = 0
#     for zipp in lstZip :
#         dfSubset = dfPolicies[dfPolicies['zip']==zipp]
#         lstBin = list(dfSubset['binSFHA'].unique())
#         intMonthCounter = 1
#         dfNew = DataFrame()
#         while intMonthCounter <= intEndCode :
#             for binn in lstBin : 
#                 dictNewRow = {}
#                 dictNewRow['year']          = intStartYear + floor((intMonthCounter)/12)
#                 dictNewRow['month']         = month_modulo(intMonthCounter)
#                 dictNewRow['zip']           = zipp
#                 dictNewRow['binSFHA']       = binn
#                 #
#                 #
#                 for var in lstConcat :
#                     dictNewRow[var] = list(dfSubset[(dfSubset['startCode']<=intMonthCounter)&
#                                                     (dfSubset['endCode']>=intMonthCounter)
#                                                     ][var].unique())
#                 #
#                 for var in lstAverage :
#                     dictNewRow[var] = dfSubset[(dfSubset['startCode']<=intMonthCounter)&
#                                                (dfSubset['endCode']>=intMonthCounter)
#                                                ][var].mean()
#                 #
#                 for var in lstSum :
#                     dictNewRow[var] = dfSubset[(dfSubset['startCode']<=intMonthCounter)&
#                                                (dfSubset['endCode']>=intMonthCounter)
#                                                ][var].sum()
#                 #
#                 dfNew = dfNew.append(dictNewRow,ignore_index=True)   
#             #
#             intMonthCounter+=1
#         #
#         dfNew.to_csv('data/working/policies.csv',index=None,mode='a',
#                      header=False)
#         intIterCount += 1
#         print('Finished Zip Code: '+str(int(zipp))+' ('+str(intIterCount)
#               +' / '+str(intIterTotal)+') '+str(dictNewRow['reportedCity']))
#     #
#     print('Single Thread: '+programTimer(timeLoop))  
# =============================================================================
    # ### Version 1
    # timeLoop = time.time()
    # #   Set new DF
    # dfNew = DataFrame()
    
    # #   Cycle through Year, Month, Zip, binSFHA for Row Consolidation
    # print('Variable Tranformation Complete ; Beginning Row Consolidation Now')
    # lstYear = list(dfClaims['year'].unique())
    # for year in lstYear :
    #     print('Starting year '+str(year))
    #     lstMonth = list(dfClaims[(dfClaims['year']==year)
    #                              ]['month'].unique())
    #     #
    #     for month in lstMonth :
    #         print('Starting month '+str(month))
    #         lstZip = list(dfClaims[(dfClaims['year']==year)
    #                                &(dfClaims['month']==month)
    #                                ]['zip'].unique())
    #         #
    #         for zipp in lstZip :
    #             lstBin = list(dfClaims[(dfClaims['year']==year)
    #                                    &(dfClaims['month']==month)
    #                                    &(dfClaims['zip']==zipp)
    #                                    ]['binSFHA'].unique())
    #             #
    #             for binn in lstBin : 
    #                 dictNewRow = {}
    #                 dictNewRow['year']          = year
    #                 dictNewRow['month']         = month
    #                 dictNewRow['zip']           = zipp
    #                 dictNewRow['binSFHA']  = binn
    #                 #
    #                 dfSubset = dfClaims[(dfClaims['year']==year)
    #                                     &(dfClaims['month']==month)
    #                                     &(dfClaims['zip']==zipp)
    #                                     &(dfClaims['binSFHA']==binn)]
    #                 #
    #                 for var in lstConcat :
    #                     dictNewRow[var] = list(dfSubset[var].unique())
    #                 #
    #                 for var in lstAverage :
    #                     dictNewRow[var] = dfSubset[var].mean()
    #                 #
    #                 for var in lstSum :
    #                     dictNewRow[var] = dfSubset[var].sum()
    #                 #
    #                 dfNew = dfNew.append(dictNewRow, ignore_index=True)
    #             #
    #         #
    #     #
    # #
    # print(programTimer(timeLoop))
# =============================================================================
# # =============================================================================    
#     df = DataFrame()
#     lstYear = list(dfClaims['year'].unique())
#     for year in lstYear :
#         #print('Starting year '+str(year))
#         lstMonth = list(dfClaims[(dfClaims['year']==year)
#                                  ]['month'].unique())
#         #
#         for month in lstMonth :
#             #print('Starting month '+str(month))
#             lstZip = list(dfClaims[(dfClaims['year']==year)
#                                    &(dfClaims['month']==month)
#                                    ]['zip'].unique())
#             #
#             for zipp in lstZip :
#                 lstBin = list(dfClaims[(dfClaims['year']==year)
#                                        &(dfClaims['month']==month)
#                                        &(dfClaims['zip']==zipp)
#                                        ]['binSFHA'].unique())
#                 #
#                 for binn in lstBin : 
#                     dictNewRow = {}
#                     dictNewRow['year']          = year
#                     dictNewRow['month']         = month
#                     dictNewRow['zip']           = zipp
#                     dictNewRow['binSFHA']       = binn
#                     #
#                     dfSubset = dfClaims[(dfClaims['year']==year)
#                                         &(dfClaims['month']==month)
#                                         &(dfClaims['zip']==zipp)
#                                         &(dfClaims['binSFHA']==binn)]
#                     #
#                     for var in lstConcat :
#                         dictNewRow[var] = list(dfSubset[var].unique())
#                     #
#                     df = df.append(dictNewRow, ignore_index=True)
#                 #
#                 print(str(zipp)+' '+str(month)+' '+str(year))
#             #
#         #
#     #
# 
# =============================================================================
# =============================================================================
#     #   Concat
#     df = DataFrame()
#     lstYear = list(dfNew['year'].unique())
#     for year in lstYear :
#         #print('Starting year '+str(year))
#         lstMonth = list(dfNew[(dfNew['year']==year)
#                                  ]['month'].unique())
#         #
#         for month in lstMonth :
#             #print('Starting month '+str(month))
#             lstZip = list(dfNew[(dfNew['year']==year)
#                                    &(dfNew['month']==month)
#                                    ]['zip'].unique())
#             #
#             for zipp in lstZip :
#                 lstBin = list(dfNew[(dfNew['year']==year)
#                                        &(dfNew['month']==month)
#                                        &(dfNew['zip']==zipp)
#                                        ]['binSFHA'].unique())
#                 #
#                 for binn in lstBin : 
#                     dictNewRow = {}
#                     dictNewRow['year']          = year
#                     dictNewRow['month']         = month
#                     dictNewRow['zip']           = zipp
#                     dictNewRow['binSFHA']       = binn
#                     #
#                     dfSubset = dfNew[(dfNew['year']==year)
#                                         &(dfNew['month']==month)
#                                         &(dfNew['zip']==zipp)
#                                         &(dfNew['binSFHA']==binn)]
#                     #
#                     for var in lstConcat :
#                         dictNewRow[var] = list(dfSubset[var].unique())
#                     #
#                     df = df.append(dictNewRow, ignore_index=True)
#                 #print(str(zipp)+' '+str(month)+' '+str(year))
#                 #
#             #
#         #
#     #
# 
# =============================================================================







