# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:28:06 2020
@author: RC

Input files:        data/working/hurricanePath.csv,
                    data/working/Production_collapsed.csv,
                    data/censusMaster.csv,
                    data/working/takeupMonthly.csv
Preceeding Script:  
Output files:       data/working/sampleStruc.csv,
                    data/working/sampleUnstr.csv
Following Script:   

This is DEPRECATED
This code takes the production, hurricane and takeup rate data, and combines
them to produce the Structured and Unstructured samples.

"""



    






# =========================================================================== #
from pandas import *

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
               ('abrState','VA','abrCounty','Hampton','Hampton City')
               ]

# =========================================================================== #










# =========================================================================== #
def merge_track (left,right,by,how='outer',override=0) :
    #   Fixed version!
    left['fromSource'] = 1
    right['fromMerger'] = 1
    
    if (how=='left')&(len(right[right[by].duplicated()])>0)&(override==0) :
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
    df = df.drop_duplicates()
    return list(df.itertuples(index=False,name=None))
####
def remove_vars (df,listVar) :
    for var in listVar :
        del df[var]
    return df
####
def get_effected_months (df) :
    listOut = []
    dfOut = DataFrame()
    bRecovery = 0
    intMonthCounter = 1
    previousProduction = 0
    previousMonth = 0
    for index,row in df.iterrows() :
        dictRow = {}
        # First row
        if (notnull(row['name']))&(bRecovery==0) :
            dictRow = dict(row)
            dictRow['prevProd'] = previousProduction
            
            dictRow['recoverCount'] = intMonthCounter
            
            previousMonth = row['month']
            
            bRecovery = 1
            dfOut = dfOut.append(dictRow,ignore_index=True)
        ####
        # The Hyde-county exception : two storms in one month
        # Middle rows
        elif ((bRecovery==1)&(row['Production']<previousProduction))|(
                (bRecovery==1)&(previousMonth==row['month'])) :
            dictRow = dict(row)
            dictRow['prevProd'] = previousProduction
            
            if previousMonth==row['month'] : dictRow['recoverCount'] = intMonthCounter
            else :
                intMonthCounter += 1
                dictRow['recoverCount'] = intMonthCounter
            ####
            
            previousMonth = row['month']
            dfOut = dfOut.append(dictRow,ignore_index=True)
        ####
        # Last row
        elif (bRecovery==1)&(row['Production']>=previousProduction) :
            dictRow = dict(row)
            dictRow['prevProd'] = previousProduction
            
            intMonthCounter += 1
            dictRow['recoverCount'] = intMonthCounter
            intMonthCounter = 1
            
            previousMonth = row['month']
            bRecovery = 0
            dfOut = dfOut.append(dictRow,ignore_index=True)
            listOut.append(dfOut)
            dfOut = DataFrame()
        ####
        # Not included row
        else : previousProduction = row['Production']
    ####
    return listOut
####
def df_to_struc_unstruc (frame) :
    dfOut = DataFrame()
    inSeasonCounter = 1
    inMonthCounter = 1
    monthOfStorm = 0
    intRecoverCount = frame['recoverCount'].max()
    for index,row in frame.iterrows() :
        if notnull(row['name']) :
            dictRow = dict(row)
            
            dictRow['recovery'] = intRecoverCount - row['recoverCount'] + 1
            
            dictRow['numInSeason'] = inSeasonCounter
            inSeasonCounter += 1
            
            if monthOfStorm==0 :
                monthOfStorm = row['month']
                dictRow['numInMonth'] = inMonthCounter
                inMonthCounter += 1
            elif monthOfStorm==row['month'] :
                dictRow['numInMonth'] = inMonthCounter
                inMonthCounter += 1
            else :
                inMonthCounter =1
                dictRow['numInMonth'] = inMonthCounter
                inMonthCounter +=1
            ####
            
            dfOut = dfOut.append(dictRow,True)
        #### end of if
    #### end of for
    frame = frame.merge(dfOut[['name','recovery','numInMonth','numInSeason']],
                            on='name',how='left')
    return (dfOut,frame)
####
def unstructured_to_unique (frame) :
    listCopyVars = ['CAT','mslp','name','time','vmax','recovery','numInMonth',
                    'numInSeason']
    dictCopy = {}
    listOut = []
    listStorms = list(frame['name'][frame['name'].notnull()].unique())
    for storm in listStorms :
        bRecord = 0
        df = DataFrame()
        for index,row in frame.iterrows() :
            if row['name'] == storm :
                bRecord = 1
                dictRow = dict(row)
                dictCopy = dict(row[listCopyVars])
                df = df.append(dictRow,True)
            elif bRecord==1 :
                dictRow = dict(row)
                for var in dictCopy.keys() :
                    dictRow[var] = dictCopy[var]
                ####
                df = df.append(dictRow,True)
            #   end if
        #   end for
        df['recoverCount'] = df['recoverCount'] - df['recoverCount'].min() + 1
        listOut.append(df)
    #   end for
    return listOut
####
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    print(__doc__)
    #   Data
    dfHurr = read_csv('data/working/hurricanePath.csv')
    dfProd = read_csv('data/working/Production_collapsed.csv')
    dfCensus    = read_csv('data/censusMaster.csv')
    dfTakeup    = read_csv('data/working/takeupMonthly.csv')
    
    
    
    
    
    
    
    
    
    
    #   Hurricane Data
    # Get abrState from 'location'
    dfHurr['abrState'] = dfHurr['location'].apply(grab_end,args=(', ',))
    
    # Get abrCounty from 'location'
    dfHurr['abrCounty'] = dfHurr['location'].apply(isolate_better,args=('',','))
    del dfHurr['location']
    
    # Subset by state
    dfHurr = subset_by_state(dfHurr,listStates)
    
    # Fix county names
    dfHurr = rename_entries(dfHurr,replaceHurr)
    
    
    
    
    
    
    
    
    
    
    #   Get unique county list
    # Get unique
    dfCensus = dfCensus.drop_duplicates(subset=['county','state'])
    dfCensus = dfCensus[['state','county','abrCounty','abrState']]
    
    # Subset by state
    dfCensus = subset_by_state(dfCensus,listStates)
    
    # Merge 
    dfHurr = merge_track(dfHurr,dfCensus,['abrState','abrCounty'],'left')[0]
    print(dfHurr[['abrState','abrCounty']][dfHurr['MergeSuccess']==0])
    dfHurr = dfHurr[['year','month','abrState','county','CAT','vmax','mslp','time','name']]
    
    print_unique(dfHurr[dfHurr[['year','month','abrState','county']
                               ].duplicated(keep=False)],[
                                   'year','month','abrState','county','name'])
    
    
    


    
                                   
                                   
                                   
                                   
    #   Production data
    # Subset by states
    dfProd['month'] = dfProd['Month'].apply(month_to_int)
    dfProd['year'] = dfProd['Year']
    dfProd = dfProd[dfProd['Year']>=intStartYear]
    dfProd = subset_by_state(dfProd,listStates)
    dfSample = merge_track(dfProd,dfTakeup,['year','month','state','county'],'left')[0]
    
    
    dfSample = remove_vars(dfSample,['abrCounty','state','Latitude','Longitude',
                                     'fromSource','fromMerger','MergeSuccess',
                                     'Month','Year'])
    
    
    
    
    
    
    
    
    
    
    dfMerge = merge_track(dfSample,dfHurr,['year','month','abrState','county'],'left',1)[0]
    # dfMerge = dfMerge[['year','month','abrState','county','Production','name',
    #                     'CAT','vmax','mslp','time']]
    dfMerge = remove_vars(dfMerge,['fromMerger','fromSource','MergeSuccess'])
    print_unique(dfMerge[dfMerge['CAT'].notnull()],['abrState','county'])
    print_unique(dfHurr[dfHurr[['year','month','abrState','county']].duplicated(keep=False)],['year','month','abrState','county'])
    # Created duplicated rows for duplicated storms
    
    
    
    
    
    
    
    
    
    
    #   Generating the DV
    # Get list of storm+production counties ['county','abrState']
    bDebug = 0
    if bDebug==0 : listCounties = df_to_list(dfMerge)
    else : listCounties = [('Wayne County','PA'),('Bay County','FL'),
                           ('Hyde County','NC'),('Allegheny County','PA'),
                           ('Camden County','GA')]
    
    
    
    
    
    
    
    
    
    
    # Take dfMerge and list of effected-counties and generate list of dfs of 
    # effected months.
    listMonths = []
    for county in listCounties :
        listMonths = listMonths + get_effected_months(dfMerge[(dfMerge['county']==county[0])&
                                                              (dfMerge['abrState']==county[1])])
        print('Isolated months : '+str(county))
    #   endfor
    
    
    
    
    
    
    
    
    
    
    # Take list of dfs of effected months, convert into a tuple of the 
    # structured dfs (row/county-storm) and unclean unstructured data
    listStruc = []
    listUnstr = []
    for frame in listMonths :
        dfs = df_to_struc_unstruc(frame)
        listStruc.append(dfs[0])
        listUnstr.append(dfs[1])
    #   endfor
    
    
    
    
    
    
    
    
    
    
    # Take list of unstructured data and convert storm names into separate dfs
    listUnique = []
    for frame in listUnstr :
        listUnique = listUnique + unstructured_to_unique(frame)
    #   endfor
    
    
    
    
    
    
    
    
    
    
    dfStruc = concat(listStruc)
    dfUnstr = concat(listUnique)
    

    
    
    
    
    
    
    
    
    dfStruc = dfStruc[['year','month','county','abrState','name','CAT',
                       'recovery','numInMonth','numInSeason','recoverCount',
                       'prevProd','Production','vmax','mslp','time',
                       'takeupTotal','ratePoverty','houseMedianValue',
                       'houseTotal','sumBuildingCoverage',
                       'policyCount','rateOccupied','rateOwnerOcc',
                       'perCapitaIncome','medianIncome','protectGap']]
    dfStruc = dfStruc.sort_values(by=['abrState','county','year','month','name'])
    
    dfUnstr = dfUnstr[['year','month','county','abrState','name','CAT',
                       'recovery','numInMonth','numInSeason','recoverCount',
                       'prevProd','Production','vmax','mslp','time',
                       'takeupTotal','ratePoverty','houseMedianValue',
                       'houseTotal','sumBuildingCoverage',
                       'policyCount','rateOccupied','rateOwnerOcc',
                       'perCapitaIncome','medianIncome','protectGap']]
    dfUnstr = dfUnstr.sort_values(by=['abrState','county','year','month','name'])
    
        
    
    
    
    
    
    
    
    
    dfStruc.to_csv('data/working/sampleStruc.csv',index=None)
    dfUnstr.to_csv('data/working/sampleUnstr.csv',index=None)
    
#   end of main()
# =========================================================================== #

































