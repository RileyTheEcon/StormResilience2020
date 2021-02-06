# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:41:03 2020
@author: RC

Input files:        data/working/takeupMonthly.csv,
                    data/working/Production-filled.csv,
                    data/working/hurricanePath.csv,
Preceeding Script:  
Output files:       data/working/sampleAllCounties.csv,
                    data/working/sampleStormCounties.csv
Following Script:   


This is the CURRENT version of this file
This takes the production, takeup rate and storm data and combines them into a
single sample file. In additon, this file creates the recovery variables.

"""










# =========================================================================== #
from pandas import *
from multiprocessing import *
set_option('display.max_rows',150)
set_option('display.width',120)

listID = ['year','month','county','abrState']

intStartYear = 2010

listStates = ['TX','LA','MS','AL','FL','GA','SC','NC','VA','MD','DE','NJ','NY',
              'CT','RI','MA','NH','ME','PA']

replaceHurr = [('abrState','TX','abrCounty','DeWitt','De Witt'),
               ('abrState','LA','abrCounty','LaSalle','La Salle'),
               ('abrState','FL','abrCounty','DeSoto','De Soto'),
               ('abrState','VA','abrCounty','Newport News','Newport News City'),
               ('abrState','VA','abrCounty','Suffolk','Suffolk City'),
               ('abrState','VA','abrCounty','Norfolk','Norfolk City'),
               ('abrState','VA','abrCounty','Portsmouth','Portsmouth City'),
               ('abrState','VA','abrCounty','Hampton','Hampton City'),
               ('abrState','TN','abrCounty','DeKalb','De Kalb')
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
def isolate_better (stri , start , end, b_end = 0) :
    stri        = str(stri)
    strShort    = ''
    posStart    = 0
    posEnd      = 0

    def reverse (stri) :
        x = ""
        for i in stri :
            x = i + x
        return x
    ####
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
####
def grab_end (x,start):
    stri        = str(x)
    posStart    = 0
    #
    posStart    = stri.find(start)+len(start)
    #
    return stri[posStart:]
####
def subset_by_state (df,listStates) :
    listdfs = []
    for state in listStates :
        listdfs.append(df[df['abrState']==state])
    ####
    df = concat(listdfs)
    return df
####
def rename_entries (df,listReplace) :
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1])&
               (df[replace[2]]==replace[3]),
               replace[2]] = replace[4]
    ####
    return df
####
def print_unique(df,listVars) :
    df = df[listVars].drop_duplicates(subset=listVars)
    df = df.sort_values(by=listVars)
    print(df)
####
def month_to_int (x) :
    if x=='January' : x = 1
    elif x=='February' : x = 2
    elif x=='March' : x = 3
    elif x=='April' : x = 4
    elif x=='May' : x = 5
    elif x=='June' : x = 6
    elif x=='July' : x = 7
    elif x=='August' : x = 8
    elif x=='September' : x = 9
    elif x=='October' : x = 10
    elif x=='November' : x = 11
    elif x=='December' : x = 12
    return x
####
def df_to_list (df) :
    df = df[['county','abrState']][df['name'].notnull()]
    return list(set(df.itertuples(index=False,name=None)))
####
def remove_vars (df,listVar) :
    for var in listVar :
        del df[var]
    return df
####
# def get_effected_months (df) :
#     listOut = []
#     dfOut = DataFrame()
#     bRecovery = 0
#     intMonthCounter = 1
#     previousProduction = 0
#     previousMonth = 0
#     for index,row in df.iterrows() :
#         dictRow = {}
#         # First row
#         if (notnull(row['name']))&(bRecovery==0) :
#             dictRow = dict(row)
#             dictRow['prevProd'] = previousProduction
            
#             dictRow['recoverCount'] = intMonthCounter
            
#             previousMonth = row['month']
            
#             bRecovery = 1
#             dfOut = dfOut.append(dictRow,ignore_index=True)
#         ####
#         # The Hyde-county exception : two storms in one month
#         # Middle rows
#         elif ((bRecovery==1)&(row['Production']<previousProduction))|(
#                 (bRecovery==1)&(previousMonth==row['month'])) :
#             dictRow = dict(row)
#             dictRow['prevProd'] = previousProduction
            
#             if previousMonth==row['month'] : dictRow['recoverCount'] = intMonthCounter
#             else :
#                 intMonthCounter += 1
#                 dictRow['recoverCount'] = intMonthCounter
#             ####
            
#             previousMonth = row['month']
#             dfOut = dfOut.append(dictRow,ignore_index=True)
#         ####
#         # Last row
#         elif (bRecovery==1)&(row['Production']>=previousProduction) :
#             dictRow = dict(row)
#             dictRow['prevProd'] = previousProduction
            
#             intMonthCounter += 1
#             dictRow['recoverCount'] = intMonthCounter
#             intMonthCounter = 1
            
#             previousMonth = row['month']
#             bRecovery = 0
#             dfOut = dfOut.append(dictRow,ignore_index=True)
#             listOut.append(dfOut)
#             dfOut = DataFrame()
#         ####
#         # Not included row
#         else : previousProduction = row['Production']
#     ####
#     return listOut
####
def get_recover_count (dfInput) :
    dfNoRecover = DataFrame()
    listRecover = []
    bRecovery = 0
    intMonthCounter = 1
    intSeasonCounter = 1
    previousProduction = 0
    previousMonth = 0
    bNoDrop = 0
    df = DataFrame()
    for index,row in dfInput.iterrows() :
        dictRow = dict(row)
        #dictRow['numInMonth'] = 1
        # First row
        if (notnull(row['name']))&(bRecovery==0)&(notnull(row['Production'])) :
            dictRow['prevProd'] = previousProduction
            dictRow['recoverCount'] = intMonthCounter
            dictRow['numInRecovery'] = intSeasonCounter
            dictRow['numInMonth'] = 1
            intMonthCounter += 1
            
            bNoDrop = int(row['Production']>previousProduction)
            
            bRecovery = 1
            dictRow['bRecovery'] = bRecovery
            
            df = df.append(dictRow,ignore_index=True)
        ####
        # Special condition : No production drop in months 1 or 2 
        #   => record 2 as non-storm/non-recovery month
        elif ((intMonthCounter==2)&(row['Production']>previousProduction)&
              (bNoDrop==1)&(bRecovery==1)) :
            listRecover.append(df)
            df = DataFrame()
            
            bRecovery = 0
            intMonthCounter = 1
            
            previousProduction = row['Production']
            dfNoRecover = dfNoRecover.append(dictRow,True)
        # Special condition : no-production counties => run first and last row
        elif (notnull(row['name']))&(isna(row['Production'])) : 
            dictRow['recoverCount'] = intMonthCounter
            dictRow['numInRecovery'] = intSeasonCounter
            dictRow['numInMonth'] = 1
            dictRow['bRecovery'] = bRecovery
            
            df = df.append(dictRow,ignore_index=True)
            listRecover.append(df)
            df = DataFrame()
        # Middle rows
        elif ((bRecovery==1)&(row['Production']<previousProduction)) :
            dictRow['prevProd'] = previousProduction
            
            dictRow['recoverCount'] = intMonthCounter
            intMonthCounter += 1
            
            if notnull(row['name']) : intSeasonCounter+=1
            dictRow['numInRecovery'] = intSeasonCounter
            
            dictRow['numInMonth'] = 1
         
            dictRow['bRecovery'] = bRecovery
         
            df = df.append(dictRow,ignore_index=True)
        ####
        # Last row
        # + Special condition : Drop in month 1 & recovery in month 2 => resolve normally
        elif ((bRecovery==1)&(row['Production']>=previousProduction))|(
                (intMonthCounter==2)&(row['Production']>previousProduction)&
                (bNoDrop==0)&(bRecovery==1)) :
            dictRow['prevProd'] = previousProduction
            
            dictRow['recoverCount'] = intMonthCounter
            intMonthCounter = 1
            
            if notnull(row['name']) : intSeasonCounter+=1
            dictRow['numInRecovery'] = intSeasonCounter
            intSeasonCounter = 1
            
            dictRow['numInMonth'] = 1
            
            dictRow['bRecovery'] = bRecovery
            bRecovery = 0
            df = df.append(dictRow,ignore_index=True)
            listRecover.append(df)
            df = DataFrame()
        ####
        # Not included row
        else : 
            previousProduction = row['Production']
            dfNoRecover = dfNoRecover.append(dictRow,True)
        #   end if
    #   end iterrow
    if len(df)>0 : listRecover.append(df)
    ####
    return dfNoRecover, listRecover
####
# def df_to_struc_unstruc (frame) :
#     dfOut = DataFrame()
#     inSeasonCounter = 1
#     inMonthCounter = 1
#     monthOfStorm = 0
#     intRecoverCount = frame['recoverCount'].max()
#     for index,row in frame.iterrows() :
#         if notnull(row['name']) :
#             dictRow = dict(row)
            
#             dictRow['recovery'] = intRecoverCount - row['recoverCount'] + 1
            
#             dictRow['numInSeason'] = inSeasonCounter
#             inSeasonCounter += 1
            
#             if monthOfStorm==0 :
#                 monthOfStorm = row['month']
#                 dictRow['numInMonth'] = inMonthCounter
#                 inMonthCounter += 1
#             elif monthOfStorm==row['month'] :
#                 dictRow['numInMonth'] = inMonthCounter
#                 inMonthCounter += 1
#             else :
#                 inMonthCounter =1
#                 dictRow['numInMonth'] = inMonthCounter
#                 inMonthCounter +=1
#             ####
            
#             dfOut = dfOut.append(dictRow,True)
#         #### end of if
#     #### end of for
#     frame = frame.merge(dfOut[['name','recovery','numInMonth','numInSeason']],
#                             on='name',how='left')
#     return (dfOut,frame)
####
def df_add_storms (frame) :
    dfOut = DataFrame()
    frame['recovery'] = frame['recoverCount'].max()
    frame['prodPerc'] = (frame['prevProd']-frame['Production'])/frame['prevProd']
    frame['prodPerc'] = frame['prodPerc'].apply(lambda x : 0 if x<0 else x)
    frame['percentScore'] = frame['prodPerc'].sum()
    s1_name,s1_CAT,s1_vmax,s1_mslp,s1_time = 0,0,0,0,0
    s2_name,s2_CAT,s2_vmax,s2_mslp,s2_time = 0,0,0,0,0
    for index,row in frame.iterrows() :
        dictRow = dict(row)
        for entry in ['name','CAT','vmax','mslp','time'] :
            del dictRow[entry]
        #   end for
        if notnull(row['name'])&(row['numInRecovery']==1) :
            s1_name = row['name']
            s1_CAT  = row['CAT']
            s1_vmax = row['vmax']
            s1_mslp = row['mslp']
            s1_time = row['time']
        if notnull(row['name'])&(row['numInRecovery']==2) :
            s2_name = row['name']
            s2_CAT  = row['CAT']
            s2_vmax = row['vmax']
            s2_mslp = row['mslp']
            s2_time = row['time']
        if notnull(row['name'])&(row['numInRecovery']==3) :
            s3_name = row['name']
            s3_CAT  = row['CAT']
            s3_vmax = row['vmax']
            s3_mslp = row['mslp']
            s3_time = row['time']
        #   end if
        dictRow['s1_name'] = s1_name
        dictRow['s1_CAT']  = s1_CAT
        dictRow['s1_vmax'] = s1_vmax
        dictRow['s1_mslp'] = s1_mslp
        dictRow['s1_time'] = s1_time
        if row['numInRecovery']==2 :
            dictRow['s2_name'] = s2_name
            dictRow['s2_CAT']  = s2_CAT
            dictRow['s2_vmax'] = s2_vmax
            dictRow['s2_mslp'] = s2_mslp
            dictRow['s2_time'] = s2_time
        elif row['numInRecovery']==3 :
            dictRow['s3_name'] = s3_name
            dictRow['s3_CAT']  = s3_CAT
            dictRow['s3_vmax'] = s3_vmax
            dictRow['s3_mslp'] = s3_mslp
            dictRow['s3_time'] = s3_time
        #   end if
        dfOut = dfOut.append(dictRow,True)
    #   end inter
    return dfOut
####
# def unstructured_to_unique (frame) :
#     listCopyVars = ['CAT','mslp','name','time','vmax','recovery','numInMonth',
#                     'numInSeason']
#     dictCopy = {}
#     listOut = []
#     listStorms = list(frame['name'][frame['name'].notnull()].unique())
#     for storm in listStorms :
#         bRecord = 0
#         df = DataFrame()
#         for index,row in frame.iterrows() :
#             if row['name'] == storm :
#                 bRecord = 1
#                 dictRow = dict(row)
#                 dictCopy = dict(row[listCopyVars])
#                 df = df.append(dictRow,True)
#             elif bRecord==1 :
#                 dictRow = dict(row)
#                 for var in dictCopy.keys() :
#                     dictRow[var] = dictCopy[var]
#                 ####
#                 df = df.append(dictRow,True)
#             #   end if
#         #   end for
#         df['recoverCount'] = df['recoverCount'] - df['recoverCount'].min() + 1
#         listOut.append(df)
#     #   end for
#     return listOut
####
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    print(__doc__)
    #   Data
    # import takeup. takeup has full year+month+state+county stem
    dfTakeup    = read_csv('data/working/takeupMonthly.csv')
    dfTakeup    = dfTakeup[dfTakeup['year']<2020]
    dfCensus    = read_csv('data/censusMaster.csv')
    dfStates    = dfCensus[['state','abrState']].drop_duplicates(['state','abrState'])
    dfTakeup    = dfTakeup.merge(dfStates,how='left',on='state')
    
    assert len(dfTakeup[dfTakeup[listID].duplicated()])==0
    assert len(dfTakeup)==377040
    
    
   
    
   
    
   
    
   
    
    #   Production data
    # Subset by states
    dfProd = read_csv('data/working/Production-filled.csv')
    
    
    
    
    
    
    
    
    
    
    #   Add production to stem in takeup to gen sample.
    #dfSample = merge_track(dfProd,dfTakeup,['year','month','state','county'],)[0]
    dfSample = merge_track(dfTakeup,dfProd,['year','month','state','county'],'left')[0]
    dfSample = remove_vars(dfSample,['state','fromSource',
                                     'fromMerger','MergeSuccess'])
    
    assert len(dfSample[dfSample[listID].duplicated()])==0
    assert len(dfSample)==377040
    
    
    
    
    
    
    
    
    
    
    #   Hurricane Data
    dfHurr = read_csv('data/working/hurricanePath.csv')
    
    # Get abrState from 'location'
    dfHurr['abrState'] = dfHurr['location'].apply(grab_end,args=(', ',))
    
    # Get abrCounty from 'location'
    dfHurr['abrCounty'] = dfHurr['location'].apply(isolate_better,args=('',','))
    del dfHurr['location']
    
    # Fix county names
    dfHurr = rename_entries(dfHurr,replaceHurr)
    
    #   Get unique county list
    # Get unique
    dfCensus = dfCensus.drop_duplicates(subset=['county','state'])
    dfCensus = dfCensus[['state','county','abrCounty','abrState']]
    
    # Merge 
    dfHurr = merge_track(dfHurr,dfCensus,['abrState','abrCounty'],'left')[0]
    print(dfHurr[['abrState','abrCounty']][dfHurr['MergeSuccess']==0])
    dfHurr = dfHurr[['year','month','abrState','county','CAT','vmax',
                     'mslp','time','name','labelPoint']]
    
    print_unique(dfHurr[dfHurr[['year','month','abrState','county']
                               ].duplicated(keep=False)],[
                                   'year','month','abrState','county','name',
                                   'CAT','vmax','mslp','time'])
    print(len(dfHurr))
    dfDouble = dfHurr[dfHurr[['county','abrState','year','month']].duplicated(keep='first')]
    print(len(dfDouble))
    dfHurr = dfHurr[~dfHurr[['county','abrState','year','month']].duplicated(keep='first')]
    print(len(dfHurr))



    
    
    
    
    
    
    
    # Add hurricane data to sample stem
    dfSample = merge_track(dfSample,dfHurr,['year','month','abrState','county'],'left')[0]
    dfSample = remove_vars(dfSample,['fromMerger','fromSource','MergeSuccess'])
    print_unique(dfSample[dfSample['CAT'].notnull()],['abrState','county'])
    print_unique(dfHurr[dfHurr[['year','month','abrState','county']].duplicated(keep=False)],['year','month','abrState','county'])
    # Created duplicated rows for duplicated storms
    
    assert len(dfSample[dfSample[listID].duplicated()])==0
    assert len(dfSample)==377040
    
    
    
    
    
    
    
    
    
    
    #   Generating the DV
    # Get list of storm+production counties ['county','abrState']
    #dfSample = subset_by_state(dfSample,listStates)
    bDebug = 0
    if bDebug==0 : listCounties = df_to_list(dfSample)
    else : listCounties = [('Wayne County','PA'),('Bay County','FL'),
                           ('Hyde County','NC'),('Allegheny County','PA'),
                           ('Camden County','GA')]
    
    
    
    
    
    
    
    
    
    
    # Add Binaries + Create list of subsets [('county','abrState')]
    dfSample['bStorm']   = dfSample['name'].apply(lambda x : 1 if notnull(x) else 0)
    dfSample['bCoastal'] = dfSample['abrState'].apply(lambda x : 1 if x in listStates else 0)
    dfSample['bEffectedCounty'] = 0
    listDfs = []
    for county in listCounties :
        dfSample.loc[(dfSample['county']==county[0])&
                     (dfSample['abrState']==county[1]),
                     'bEffectedCounty'] = 1
        listDfs.append(dfSample[(dfSample['county']==county[0])&
                                (dfSample['abrState']==county[1])])
    #   end for
    dfNoStorm = dfSample[dfSample['bEffectedCounty']==0]
    intBEffected = dfSample['bEffectedCounty'].sum()
    assert len(concat(listDfs+[dfNoStorm]))==377040
    
    
    
    
    
    
    
    
    
    
    for df in listDfs :
        if ('Tishomingo County' in list(df['county']))|('Hyde County' in list(df['county'])) :
            print(df.head(10))
        #   end
    #   end
    

    
    
    
    
    
    
    
    
    # iter thru dfs get_recover_count(df)
    # return dfNoRecover, listRecover
    # pool = Pool(cpu_count())    
    # listResult = pool.map(generate_stem,listStem)
    dfNoRecover = DataFrame()
    listRecover = []
    for df in listDfs :
        result = get_recover_count(df)
        dfNoRecover = dfNoRecover.append(result[0],True)
        listRecover = listRecover + result[1]
    #   end for
    
    assert len(concat([dfNoRecover]+listRecover+[dfNoStorm]))==377040
    
    
    
    
    
    
    
    
    
    
    # for df in listRecover :
    #     print(df[['month','county','abrState','name','numInMonth','numInSeason','recoverCount']])
    for df in listRecover :
        if ('Tishomingo County' in list(df['county']))|('Hyde County' in list(df['county'])) :
            print(df[['month','county','abrState','name','numInMonth','recoverCount']])
        #   end
    #   end
    
    
    
    
    
    
    
    
    
    
    intStormMax = 0
    for df in listRecover :
        if intStormMax < df['numInRecovery'].max() : intStormMax = df['numInRecovery'].max()
    #   end
    print(intStormMax)
    
    
    
    
    
    
    
    
    
    
    listComplete = []
    for df in listRecover:
        listComplete.append(df_add_storms(df))
    #   end for
    dfRes = concat([dfNoRecover]+listComplete)
    assert len(concat([dfRes,dfNoStorm]))==377040
    assert len(dfRes[dfRes[listID].duplicated()])==0
    
    
    
    
    
    
    
    
    
    
    dfRes['numInSeason'] = 0
    listTuples = list(set(dfRes[['year','abrState','county']][
        (dfRes['bStorm']==1)].itertuples(index=False,name=None)))
    
    for tpl in listTuples :
        intStormCounter = 0
        for index,row in dfRes[(dfRes['year']==tpl[0])&(dfRes['abrState']==tpl[1])
                               &(dfRes['county']==tpl[2])&(dfRes['bRecovery']==1)].iterrows() :
            if row['bStorm']==1 : intStormCounter+=1
            dfRes.loc[(dfRes['year']==tpl[0])&(dfRes['month']==row['month'])&
                      (dfRes['abrState']==tpl[1])&(dfRes['county']==tpl[2]),
                      'numInSeason'] = intStormCounter
    #   end for
    assert len(dfRes[dfRes[listID].duplicated()])==0
    
    
    
    
    
    
    
    
    
    
    #   These four results managed to get missed by the algorithm. 
    #   For expediency, the entries are manually edited.
    #   Tishomingo county
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              'numInSeason'] =2
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              'numInMonth'] = 2
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              'numInRecovery'] = 2
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              's2_name'] = 'irma'
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              's2_CAT'] = 'LO'
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              's2_vmax'] = 15
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              's2_mslp'] = 1004
    dfRes.loc[(dfRes['county']=='Tishomingo County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='MS'),
              's2_time'] = 1.405966235338218
    
    #   Hyde county
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              'numInSeason'] =2
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              'numInRecovery'] =2
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              'numInMonth'] = 2
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              's2_name'] = 'julia'
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              's2_CAT'] = 'EX'
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              's2_vmax'] = 25
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              's2_mslp'] = 1011
    dfRes.loc[(dfRes['county']=='Hyde County')&
              (dfRes['s1_name']=='hermine')&
              (dfRes['abrState']=='NC'),
              's2_time'] = 3.5942751088332754
    
    #   Hardin county
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              'numInSeason'] =2
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              'numInMonth'] = 2
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              'numInRecovery'] = 2
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              's2_name'] = 'irma'
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              's2_CAT'] = 'LO'
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              's2_vmax'] = 15
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              's2_mslp'] = 1004
    dfRes.loc[(dfRes['county']=='Hardin County')&
              (dfRes['s1_name']=='harvey')&
              (dfRes['abrState']=='TN'),
              's2_time'] = 0.608927
    
    
    #   Walker County back-to-back storm glitch
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              'numInSeason'] = 1
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              'numInMonth'] = 1
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              'numInRecovery'] = 1
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              's1_name'] = 'nate'
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              's1_CAT'] = 'TD'
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              's1_vmax'] = 35
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              's1_mslp'] = 995
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              's1_time'] = 1.28127901665084
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              'recovery'] = 1
    dfRes.loc[(dfRes['county']=='Walker County')&
              (dfRes['year']==2017)&
              (dfRes['month']==10)&
              (dfRes['abrState']=='AL'),
              'percentScore'] = 0
    
    #   Hardee County back-to-back storm glitch
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              'numInSeason'] = 1
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              'numInMonth'] = 1
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              'numInRecovery'] = 1
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              's1_name'] = 'irma'
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              's1_CAT'] = 'H1'
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              's1_vmax'] = 80
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              's1_mslp'] = 955
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              's1_time'] = 1.3260611622311
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              'recovery'] = 1
    dfRes.loc[(dfRes['county']=='Hardee County')&
              (dfRes['year']==2017)&
              (dfRes['month']==9)&
              (dfRes['abrState']=='FL'),
              'percentScore'] = 0
    
    
    
    
    
    
    
    
    
    
    dfRes['recovery12'] = dfRes['recovery'].apply(lambda x : 13 if x>12 else x)
    dfRes['recovery24'] = dfRes['recovery'].apply(lambda x : 25 if x>24 else x)
    
    
    
    
    
    
    
    
    
    
    dfRes = dfRes[['year','month','county','abrState',
                   'bEffectedCounty','bStorm','bCoastal','bRecovery',
                   'bProduction','Production','prevProd','distance',
                   'prodPerc','percentScore','recovery','recovery12','recovery24',
                   'numInMonth','numInSeason','numInRecovery','recoverCount',
                   's1_name','s1_CAT','s1_vmax','s1_mslp','s1_time',
                   's2_name','s2_CAT','s2_vmax','s2_mslp','s2_time',
                   's3_name','s3_CAT','s3_vmax','s3_mslp','s3_time',
                   'takeupTotal','protectGap','totalInsurableValue','insuredValue',
                   'houseMedianValue','houseTotal','medianIncome',
                   'perCapitaIncome','rateOccupied','rateOwnerOcc','ratePoverty',
                   'policyCount','sumBuildingCoverage','sumBuildingDeductib',
                   'sumContentsCoverage','sumContentsDeductib'
                   ]]
    
    dfUnres = concat([dfRes,dfNoStorm])    
    assert len(dfUnres[dfUnres[listID].duplicated()])==0
    assert len(dfUnres)==377040
    
    dfRes = subset_by_state(dfRes,listStates)
    dfRes = dfRes.sort_values(by=['abrState','county','year','month']).reset_index(drop=True)
    dfRes.to_csv('data/working/sampleStormCounties.csv',index=None)
    
    
    
    
    
    
    
    
    
    
    dfUnres = dfUnres[['year','month','county','abrState',
                       'bEffectedCounty','bStorm','bCoastal','bRecovery',
                       'bProduction','Production','prevProd','distance',
                       'prodPerc','percentScore','recovery','recovery12','recovery24',
                       'numInMonth','numInSeason','numInRecovery','recoverCount',
                       's1_name','s1_CAT','s1_vmax','s1_mslp','s1_time',
                       's2_name','s2_CAT','s2_vmax','s2_mslp','s2_time',
                       's3_name','s3_CAT','s3_vmax','s3_mslp','s3_time',
                       'takeupTotal','protectGap','totalInsurableValue','insuredValue',
                       'houseMedianValue','houseTotal','medianIncome',
                       'perCapitaIncome','rateOccupied','rateOwnerOcc','ratePoverty',
                       'policyCount','sumBuildingCoverage','sumBuildingDeductib',
                       'sumContentsCoverage','sumContentsDeductib'
                       ]]
    dfUnres = dfUnres.sort_values(by=['abrState','county','year','month']).reset_index(drop=True)
    dfUnres.to_csv('data/working/sampleAllCounties.csv',index=None)
    
#   endmain
# =========================================================================== #





















   

    
    
    
    
    

    
    
    

    

    

    
    
    
    

        
        
        
        

    
    
    
    
    
    
    





























