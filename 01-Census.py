# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 12:36:06 2020
@author: RC


Input files:        data/source/ZipAmericanCommunitySurvey/econ+county/ACSDP5Y##.DP03_data_with_overlays_2020-11-05T152552.csv
                    data/source/ZipAmericanCommunitySurvey/house+county/ACSDP5Y##.DP04_data_with_overlays_2020-11-08T040054.csv
Preceeding Script:  00-Fip Master.py
Output files:       data/working/censusACSEconCounty.csv,
                    data/working/censusACSHouseCounty.csv,
                    data/working/censusACS.csv
Following Script:   

Year 2018 needs to be converted from ANSI to UTF-8
This takes the annaul American Community Survey for demographic and 
economic data at the county-level. It reads through a list of file names, 
drops unused variables, and combines into single file.

"""










# ==============================Packages======================================= 
from pandas import *
# =============================================================================










# ==========================Globals============================================
listYear = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
dictEmp  = {'ZIP'       :1,
            'NAME'      :1,
            'EMPFLAG'   :0,
            'EMP'       :1,
            'QP1'       :1,
            'AP'        :1,
            'EST'       :0
            }
dictEmpRename   = {'zip':'ZIP',
                   'name':'NAME',
                   'emp':'EMP',
                   'qp1':'QP1',
                   'ap':'AP',
                   'est':'EST'
                   }
dfEmpMaster = DataFrame()
# =============================================================================










# ===========================Functions=========================================
def remove_unused_vars (df,dictVars) :
    #   Get drop list
    intColumns = len(list(df))
    lstDropVars = [var for var,value in dictVars.items() if value!=0]
    
    #   Drop vars
    df = df[lstDropVars]
    print('Number of columns dropped: '+str(intColumns-len(list(df))))
    return df
####
def rename_variables (df,dictRename) :
    #   Format : dictRename{'start':'end'}
    for keyVar in dictRename.keys() :
        if keyVar in df.columns:
            df.rename(columns={keyVar:dictRename[keyVar]},inplace=True)
        ####
    ####
    return df
####
def reverse (stri) :
    x = ""
    for i in stri :
        x = i + x
    return x
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
def sum_stats (df,variable) :
    print('Min    :'+str(df[variable].min())+'\n'
          'Mean   :'+str(df[variable].mean())+'\n'
          'Median :'+str(df[variable].median())+'\n'
          'Max    :'+str(df[variable].max())
          )
####
def add_year (df) :
    dfYear = df[df['year']==2018]
    
    dfYear['year'] = 2019
    df = df.append(dfYear,ignore_index=True)
    
    dfYear['year'] = 2020
    df = df.append(dfYear,ignore_index=True)
    
    return df
####
def fix_county_name (df) :
    listReplace = [('state','Alaska','county','Wade Hampton Census Area','Kusilvak Census Area'),
               ('state','Alaska','county','Petersburg Borough','Petersburg Census Area'),
               ('state','South Dakota','county','Shannon County','Oglala Lakota County'),
               ('state','New Mexico','county','Do?a Ana County','Dona Ana County'),
               ('state','New Mexico','county','Doña Ana County','Dona Ana County'),
               ('state','New Mexico','county','Do�a Ana County','Dona Ana County'),
               ('state','Virginia','county','Bedford city','Bedford County'),
               ('state','Louisiana','county','LaSalle Parish','La Salle Parish')
               ]
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1])&(df[replace[2]]==replace[3]),replace[2]] = replace[4]
    ####
    return df
####
# =============================================================================










# ====================================MAIN=================================== #
if __name__ == '__main__' :
    print(__doc__)

    #   Define frames / Define Vars to Keep 
    dfEconMaster = DataFrame()
    dfHouseMaster = DataFrame()
    
    dictEconRename = {'DP03_0088E':'econPerCapitaIncome',
                      'DP03_0119PE':'econPoverty',
                      'DP03_0062E':'econMedianIncome',
                      'DP03_0063E':'econMeanIncome',
                      'DP03_0051E':'econTotalHouseholds'
                      }
    
    dictHouseRename = {'DP04_0001E':'houseTotalUnits',
                       'DP04_0045E':'houseOwnerOccupied'
                       }
    
    
    
    
    
    
    
    
    
    
    #   Import Econ Data by Year
    for year in [2010,2011,2012,2013,2014,2015,2016,2017,2018] :
        dfEconCounty = read_csv('data/source/ZipAmericanCommunitySurvey/econ+county/ACSDP5Y'
                              +str(year)+'.DP03_data_with_overlays_2020-11-05T152552.csv',
                              dtype=str)
        
        print('start year '+str(year)+' '+str(len(dfEconCounty)))
        
        
        
        
        
        
        
        
        
        #   Drop first row / Rename columns / Drop Unused Vars
        dfEconCounty = rename_variables(dfEconCounty,dictEconRename)
        dfEconCounty = dfEconCounty.drop(0)
        dfEconCounty['state'] = dfEconCounty['NAME'].apply(grab_end,args=(', ',))
        dfEconCounty['county'] = dfEconCounty['NAME'].apply(isolate_better,args=('',','))
        dfEconCounty['year'] = year
        
        dfEconCounty = dfEconCounty[['GEO_ID','year','county','state','econPerCapitaIncome',
                                     'econPoverty','econMedianIncome',
                                     'econMeanIncome','econTotalHouseholds']]
        
        
        
        
        
        
        
        
        
        #   For numeric set of columns, convert to numeric / print sum stats
        listNumeric = ['year','econPerCapitaIncome','econPoverty',
                        'econMedianIncome','econMeanIncome','econTotalHouseholds']
        
        for var in listNumeric :
            dfEconCounty[var] = to_numeric(dfEconCounty[var],errors='coerce')
            print('\n'+str(year)+' : '+str(var))
            sum_stats(dfEconCounty,var)
        #   endfor
        
        
        
        
        
        
        
        
        
        #   Add to existing frame
        dfEconMaster = dfEconMaster.append(dfEconCounty,ignore_index=True)
        print()
        print(dfEconMaster['year'].value_counts().sort_index())
    #   endfor year
    
    
    
    
    
    
    
    
    
    #   Export compleyed Econ data
    dfEconMaster.to_csv('data/working/censusACSEconCounty.csv',index=None)
    
    
    
    
    
    
    
    
    
    
    #   Import housing data by year
    for year in [2010,2011,2012,2013,2014,2015,2016,2017,2018] :
        dfHouseCounty = read_csv('data/source/ZipAmericanCommunitySurvey/house+county/ACSDP5Y'
                              +str(year)+'.DP04_data_with_overlays_2020-11-08T040054.csv',
                              dtype=str)
        print('start year '+str(year)+' '+str(len(dfHouseCounty)))
    
    
        
        
        
        
        
        
        
        
        #   For years <2015 , houseMedianValue in different column
        if year<2015 : dictHouseRename['DP04_0088E'] = 'houseMedianValue'
        else : dictHouseRename['DP04_0089E'] = 'houseMedianValue'
        
        
        
        
        
        
        
        
        
        
        #   Drop first row / Rename columns / Drop Unused Vars
        dfHouseCounty = rename_variables(dfHouseCounty,dictHouseRename)
        dfHouseCounty = dfHouseCounty.drop(0)       # drop first row (second set of titles)
        dfHouseCounty['state'] = dfHouseCounty['NAME'].apply(grab_end,args=(', ',))
        dfHouseCounty['county'] = dfHouseCounty['NAME'].apply(isolate_better,args=('',','))
        dfHouseCounty['year'] = year
        
        dfHouseCounty = dfHouseCounty[['GEO_ID','year','county','state','houseTotalUnits',
                                       'houseOwnerOccupied','houseMedianValue']]
    
    
        
        
        
        
        
        
        
        
        #   For numeric set of columns, convert to numeric / print sum stats
        listNumeric =['year','houseTotalUnits','houseOwnerOccupied','houseMedianValue']
        for var in listNumeric :
            dfHouseCounty[var] = to_numeric(dfHouseCounty[var],errors='coerce')
            print('\n'+str(year)+' : '+str(var))
            sum_stats(dfHouseCounty,var)
        #   endfor
    
        
    
    
    
    
    
    
    
    
        #   Add to frame
        dfHouseMaster = dfHouseMaster.append(dfHouseCounty,ignore_index=True)
        print()
        print(dfHouseMaster['year'].value_counts().sort_index())
    #   endfor
    
    
    
    
    
    
    
    
    
    
    #   Export Housing frame
    dfHouseMaster.to_csv('data/working/censusACSHouseCounty.csv',index=None)
    
    
    
    
    
    
    
    
    
    
    #   Combine housing + econ frames / correct county names to standard / export
    dfACS = merge_track(dfHouseMaster,dfEconMaster,['year','GEO_ID','county','state'])[0]
    dfACS = dfACS.drop(columns=['fromSource','fromMerger','MergeSuccess'])
    dfACS = add_year(dfACS)
    dfACS = fix_county_name(dfACS)
    
    
    
    
    
    
    
    
    
    
    #   Export final file
    dfACS.to_csv('data/working/censusACS.csv',index=None)
#   endif
# =========================================================================== #



















