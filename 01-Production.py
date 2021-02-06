# -*- coding: utf-8 -*-
"""
@author: RC

Input files:        data/source/860/####.xlsx,
                    data/source/860/####.xls,
                    data/source/923/####.xlsx,
                    data/source/923/####.xls
Preceeding Script:  
Output files:       data/working/Production.csv,
                    data/working/PlantMaster.csv
Following Script:   

This code was developed to import forms 923 and 860 from the EIA, and 
generate a dataset of electricity production at the state/county/zip/power
plant level over the years 2001 to 2019--use out side of that time period
is also possible but hasn't been tested.
Since the 923 contains production data at the generator-level, with state
and plant name info and the 860 contains more specific location data, then
we are able to merge the two datasets. From there we are able to aggregate
the production data from individual generators up to the specified level.
The dataset is transposed, saved, and appended to the total set.
Excel file names have been changed prior to code execution to aid in 
organization and automation.
Coded by Riley Conlon, NYU

"""










# =========================================================================== #
from pandas import *
options.mode.chained_assignment = None

# =========================================================================== #










# =========================================================================== #
#intRunStateTest = 0                             # ==1 => test for unique plant names within state
#lstAggregateBy  = ['State','County']            # Set grouping for production aggregation--'State','Zip','County work 
intStartYear    = 2001                          # Set first year of data, inclusive
intEndYear      = 2019                          # Set last year of data, inclusive
intRestartPlantMaster = 1
intRestartProdMaster = 1

# =========================================================================== #










# =========================================================================== #
### Define functions
##  format data, remove spreadsheet header, set column names
def format_data (df,firstRow) :
    #   Set uniform column names
    lstColumns = list(df)
    i =0
    for col in lstColumns :
        df = df.rename(columns={col:'col'+str(i)})
        i+=1
    #
    #   Find row of names; drop all above it; set col names w row; drop row of names
    intFirstRow = df[df['col0']==firstRow].index[0]
    if intFirstRow!=0: df = df.loc[intFirstRow:]
    df.columns = df.loc[intFirstRow]
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(columns=['index'])
    return df
#
##  Check for Duplicate State names across
def test_state_unique (df) : 
    lstStates = list(set(df['State'].tolist()))
    dfState = DataFrame()
    for state in lstStates :
        dfState = df[df['State']==state]
        bolDuplicate = (len(dfPlants['Plant Name'].tolist()) == len(set(dfPlants['Plant Name'].tolist())))
        print('Is dup in '+state+' == '+str(bolDuplicate))
    #
#
##  Mark incomplete entries, as decided by critical variable list 
def mark_incomplete (df,lstCritical,name) :
    df['Missing'+name] = 0
    for var in lstCritical :
        df.loc[df[var].isnull(),'Missing'+name] = 1
    return df
#
##  Merge but also mark sources of data for debugging. This func behaves like the
#   merge in Stata
def merge_track (left,right,by,how='outer') :
    left['fromSource'] = 1
    right['fromMerger'] = 1
    dfMerge = left.merge(right,how=how,on=by)
    
    intLeft         = len(left)
    intRight        = len(right)
    intMatch        = len(dfMerge[(dfMerge['fromSource']==1)&(dfMerge['fromMerger']==1)])
    intLeftError    = len(dfMerge[(dfMerge['fromSource']==1)&(dfMerge['fromMerger'].isnull())])
    intRightError   = len(dfMerge[(dfMerge['fromSource'].isnull())&(dfMerge['fromMerger']==1)])
    
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
##  Aggregate by group, transpose wide to long
def aggregate_transpose (df,lstVar,level) :
    df = df.groupby(level)[lstVar].sum()
    df = melt(df.reset_index(),id_vars=level,value_vars=lstVar)
    df = df.rename(columns={'variable':'Month','value':'Production'})
    return df
####
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    print(__doc__)
    ##  Start Loop through years for 860 forms
    #   I decided to start at the last year and work backwards, because the code
    #   was first developed for 2019. Thus this would hopefully minimize the number
    #   of errors and exceptions that we would encounter at one time.
    intYear         = intEndYear        # iterating variable
    dfPlantMaster   = DataFrame()       # master directory for power plants
    dfPlants        = DataFrame()       # holds given year's data, deleted after while loop to save memory and performance
    
    if intRestartPlantMaster == 0 :
        try : dfPlantMaster = read_csv('data/working/PlantMaster.csv')
        except : 
            intRestartPlantMaster = 1
            print('No previous Plant Master found. Returning to business as usual.')
        #   endtry
    #   endif
    
    
    
    
    
    
    
    
    
    #   Start loop through each year in range
    while intYear >= intStartYear :
        print('='*30)
        #   Import and continue if previous work had been done
        if intRestartPlantMaster == 0 : 
            if len(dfPlantMaster.loc[(dfPlantMaster['AsOf']==intYear)]) > 0 :
                print('Year '+str(intYear)+' Found! Going to next year.')
                intYear -= 1
                continue
            #   endif
        #   endif
    
        
        
        
        
        
        
        
        
        
        #   Import data from excel
        try     : dfPlants = read_excel(r'data/source/860/'+str(intYear)+'.xlsx')
        except  : dfPlants = read_excel(r'data/source/860/'+str(intYear)+'.xls')
        
        
        
        
        
        
        
        
        
        
        #   Format to remove header.
        #   Note: For 2001-2010, data starts in first row wo any headers
        if intYear > 2011 : dfPlants = format_data(dfPlants,'Utility ID')
        elif intYear == 2011 : dfPlants = format_data(dfPlants,'UTILITY_ID')
        
        
        
        
        
        
        
        
        
        
        #   Version control: Variable names
        #   Sets all versions of the form to use the same variable names. While
        #   pre- and post- 2010 forms mostly match, there's some specific changes
        #   in singular years. For years in which specific data is missing, blanks
        #   are filled with zeroes. Years 2019 to 2013 have standard var names.
        dfPlants['AsOf'] = intYear
        if intYear < 2013 :
            dfPlants['Latitude']        = 0
            dfPlants['Longitude']       = 0
        if intYear < 2007 :
            dfPlants['City']            = ''
            dfPlants['Street Address']  = ''
        dictRename = {'PLANT_NAME'                                  :'Plant Name',
                      'PLNTNAME'                                    :'Plant Name',
                      'OWNERTRANSDIST'                              :'Utility Name',
                      'SERVICE AREA'                                :'Utility Name',
                      'SERVAREA'                                    :'Utility Name',
                      'PLANT_CODE'                                  :'Plant Code',
                      'PLNTCODE'                                    :'Plant Code',
                      'UTILITY_ID'                                  :'Utility ID',
                      'UTILCODE'                                    :'Utility ID',
                      'STREET_ADDRESS'                              :'Street Address',
                      'MAIL_STREET_ADDRESS'                         :'Street Address',
                      'CITY'                                        :'City',
                      'MAIL_CITY'                                   :'City',
                      'STATE'                                       :'State',
                      'PLNTSTATE'                                   :'State',
                      'ZIP5'                                        :'Zip',
                      'PLNTZIP'                                     :'Zip',
                      'COUNTY'                                      :'County',
                      'CNTYNAME'                                    :'County'
                      }
        if intYear not in [2019,2018,2017,2016,2015,2014,2013] : 
            dictRename['Transmission or Distribution System Owner'] = 'Utility Name'
        #   endif
        for keyVar in dictRename.keys() :
            if keyVar in dfPlants.columns:
                dfPlants.rename(columns={keyVar:dictRename[keyVar]},inplace=True)
            #   endif
        #   endfor
        
        
        
        
        
        
        
        
        
        #   Keep Important vars
        dfPlants = dfPlants[['Plant Name','Plant Code','Utility Name','Utility ID',
                             'Street Address','City','State','Zip','County',
                             'Latitude','Longitude','AsOf']]
        print('Year: '+str(intYear)+' Vars: '+str(list(dfPlants)))
        
        
        
        
        
        
        
        
        
        
        #   Uniqueness Test
        intNewAdd = 0
        if len(dfPlantMaster) == 0 : 
            dfPlantMaster = dfPlants
            intNewAdd = len(dfPlantMaster)
        else :
            for index, row in dfPlants.iterrows() :
                #   If a plant does not exist in the master, then add it
                if len(dfPlantMaster.loc[(dfPlantMaster['State']==row['State'])
                                         &(dfPlantMaster['Plant Name']==row['Plant Name'])
                                         ]) == 0 :
                    dfPlantMaster = dfPlantMaster.append({'Plant Name'      :row['Plant Name'],
                                                          'Plant Code'      :row['Plant Code'],
                                                          'Utility Name'    :row['Utility Name'],
                                                          'Utility ID'      :row['Utility ID'],
                                                          'Street Address'  :row['Street Address'],
                                                          'City'            :row['City'],
                                                          'State'           :row['State'],
                                                          'Zip'             :row['Zip'],
                                                          'County'          :row['County'],
                                                          'Latitude'        :row['Latitude'],
                                                          'Longitude'       :row['Longitude'],
                                                          'AsOf'            :row['AsOf']}
                                                         ,ignore_index=True)
                    intNewAdd += 1
                #   end of if
            #   end of for
        #   end of else
        print('Master Plant Directory Updated for Year: '+str(intYear)+'\n'
              +'New Additions: '+str(intNewAdd)+' out of '+str(len(dfPlants)))
        intYear -= 1
    #   end of while
    del dfPlants
    
    
    
    
    
    
    
    
    
    
    #   Save results
    dfPlantMaster.to_csv('data/working/PlantMaster.csv',index=False)
    
    
    
    
    
    
    
    
    
    
    #   Report on finds
    print('Total Number of Plants Found: '+str(len(dfPlantMaster)))
    lstPlantVars = list(dfPlantMaster)
    for var in lstPlantVars :
        intBlankCount = len(dfPlantMaster[(dfPlantMaster[var]=='')
                                          |(dfPlantMaster[var]==0)
                                          |(dfPlantMaster[var]=='0')])
        print('Number of blanks in '+str(var)+': '+str(intBlankCount))
    #
    
    
    
    
    
    
    
    
    
    ##  Get 923 Production Data
    intYear         = intStartYear        # iterating variable
    dfProdMaster    = DataFrame()       # master directory for power plants
    dfProduction    = DataFrame()       # holds given year's data, deleted after while loop to save memory and performance
    intStartNew     = 1
    
    while intYear <= intEndYear :
        print('='*30+'\n'+'Starting Merge + Aggregation for '+str(intYear))    
        #   Detect if Year is Present in the data    
        #   Import Single Year Data
        try     : dfProduction = read_excel(r'data/source/923/'+str(intYear)+'.xlsx')
        except  : dfProduction = read_excel(r'data/source/923/'+str(intYear)+'.xls')
        
        #   Format
        try : dfProduction = format_data(dfProduction,'Plant Id')
        except : dfProduction = format_data(dfProduction,'Plant ID')
        
        
        
        
        
        
        
        
        
        
        #   Correct Variable Names, Drop Unused Variables
        #   Should have names: 'Plant Name', 'State', 'Fuel Type',
        #   'Netgen\nJanuary'-'December'
        if intYear in [2019,2018,2017,2016,2015,2014,2012] :
            dfProduction = dfProduction.rename(columns={'Reported\nFuel Type Code':'Fuel Code',
                                         'Plant State':'State',
                                         'Netgen\nJanuary'      :'January',
                                         'Netgen\nFebruary'     :'February',
                                         'Netgen\nMarch'        :'March',
                                         'Netgen\nApril'        :'April',
                                         'Netgen\nMay'          :'May',
                                         'Netgen\nJune'         :'June',
                                         'Netgen\nJuly'         :'July',
                                         'Netgen\nAugust'       :'August',
                                         'Netgen\nSeptember'    :'September',
                                         'Netgen\nOctober'      :'October',
                                         'Netgen\nNovember'     :'November',
                                         'Netgen\nDecember'     :'December'
                                         })
        elif intYear in [2013,2011] :
            dfProduction = dfProduction.rename(columns={'Reported Fuel Type Code':'Fuel Code',
                                         'Netgen_Jan':'January',
                                         'Netgen_Feb':'February',
                                         'Netgen_Mar':'March',
                                         'Netgen_Apr':'April',
                                         'Netgen_May':'May',
                                         'Netgen_Jun':'June',
                                         'Netgen_Jul':'July',
                                         'Netgen_Aug':'August',
                                         'Netgen_Sep':'September',
                                         'Netgen_Oct':'October',
                                         'Netgen_Nov':'November',
                                         'Netgen_Dec':'December'
                                         })
        elif intYear in [2010,2009,2008,2007,2006,2005,2004,2003,2002,2001] :
            dfProduction = dfProduction.rename(columns={'Reported Fuel Type Code':'Fuel Code',
                                                        'NETGEN_JAN':'January',
                                                        'NETGEN_FEB':'February',
                                                        'NETGEN_MAR':'March',
                                                        'NETGEN_APR':'April',
                                                        'NETGEN_MAY':'May',
                                                        'NETGEN_JUN':'June',
                                                        'NETGEN_JUL':'July',
                                                        'NETGEN_AUG':'August',
                                                        'NETGEN_SEP':'September',
                                                        'NETGEN_OCT':'October',
                                                        'NETGEN_NOV':'November',
                                                        'NETGEN_DEC':'December'})
        #   endif
        
        
        
        
        
        
        
        
        
        
        #   Merge with Plant Master
        #   Since Plant Master has info for all plants from the last 20 years, we
        #   don't want that excess info to get copied into the production data w
        #   an outer merge. Instead we use left, and exclude all unmatched plant info
        dfProduction = merge_track(dfProduction,dfPlantMaster,by=['State','Plant Name'],how='left')[0]
        
        
        
        
        
        
        
        
        
        
        #   Aggregate, Transpose, Fix variable names
        lstNetGen = ['January','February','March','April','May','June','July',
                     'August','September','October','November','December']
        lstGroupVars = ['Plant Name','Plant Code','Utility Name','Utility ID',
                        'Fuel Code','Street Address','City','State','Zip','County',
                        'Latitude','Longitude','AsOf','MergeSuccess']
        dfProduction = dfProduction[lstNetGen+lstGroupVars]
         
        
        
        
        
        
        
        
        
        
        #   Filled gaps, replace Nan, convert back to float, subset variable names
        dfProduction[lstNetGen].fillna(0,inplace=True)
        i = 0
        for var in lstNetGen :
            dfProduction.loc[dfProduction[var]=='.',var] = 0
            dfProduction[var] = dfProduction[var].astype(float)
        #
        dfProduction = aggregate_transpose(dfProduction,lstNetGen,lstGroupVars)
    
        
    
    
    
    
    
    
    
    
        #   Report Merge Stats
        intMergeSuccess = len(dfProduction[dfProduction['MergeSuccess']==1])
        intRowTotal     = len(dfProduction)
        fltMergePercent = round(100*(intMergeSuccess/intRowTotal),2)
        print('Year '+str(intYear)+' Complete!'+'\n'
              +str(intMergeSuccess)+' Successful Merges ('+str(fltMergePercent)
              +'%), '+str(intRowTotal)+' Rows Added to Total.')
        
        
        
        
        
        
        
        
        
        
        #   Append and Export
        dfProduction['Year'] = intYear
        if intRestartProdMaster == 1 : 
            dfProduction.to_csv('data/working/Production.csv',index=None)
            intRestartProdMaster = 0
        elif intRestartProdMaster == 0 : 
            dfProduction.to_csv('data/working/Production.csv',index=None,
                                header=False,mode='a')
        intYear += 1
        #
    #
#   endif
# =========================================================================== #
# 
#     #   Import 923
#     #   923 = Levels of Production by plant (only state-level info; plant names repeat)
#     dfProduction = read_excel(r'data/923/'+str(intYear)+'.xlsx')
#     dfProduction = format_data(dfProduction,'Plant Id')
#     dfProduction = dfProduction.rename(columns={'Plant State':'State'})
#     
#     
#     
#     ##  Check for duplicate plant names within state / verify state+plant name is unique
#     if intRunStateTest == 1 : test_state_unique(dfPlants)
#     
#     
#     
#     ##  Define critical variables--if a row lacks an entry for one of these vars it
#     #   will be considered incomplete and likely unusable.
#     lstCriticalPlants = ['Utility ID','Utility Name','Plant Code','Plant Name',
#                          'Street Address','City','State','Zip','County','Latitude',
#                          'Longitude']
#     lstCriticalProduction = ['Plant Id','Plant Name','State']
#     
#     dfProduction = mark_incomplete(dfProduction,lstCriticalProduction,'Production')     # adds col 'MissingProduction'
#     dfPlants = mark_incomplete(dfPlants,lstCriticalPlants,'Plants')                     # adds col 'MissingPlants'
#     
#     
#     #   Merges sets, reports mismatch, adds cols 'fromSource' 'fromMerger'
#     dfMerge = merge_track(dfProduction,dfPlants,by=['State','Plant Name'])[0]           # need index since func returns vector
#     
#     
#     
#     #   Count incomplete entries, complete entries for later reporting
#     intProductionMissing    = len(dfProduction[dfProduction['MissingProduction']==1])
#     intProduction           = len(dfProduction[dfProduction['MissingProduction']==0])
#     intPlantMissing         = len(dfPlants[dfPlants['MissingPlants']==1])
#     intPlant                = len(dfPlants[dfPlants['MissingPlants']==0])
#     
#     intPlantToProdError     = len(dfMerge[(dfMerge['MissingPlants']==0)&(dfMerge['MissingProduction']==1)])
#     intPlantMismatchError   = len(dfMerge[(dfMerge['MissingPlants']==0)&(dfMerge['MissingProduction'].isnull())])
#     intProdToPlantError     = len(dfMerge[(dfMerge['MissingPlants']==1)&(dfMerge['MissingProduction']==0)])
#     intProdMismatchError    = len(dfMerge[(dfMerge['MissingPlants'].isnull())&(dfMerge['MissingProduction']==0)])
#     intSuccessMatch         = len(dfMerge[(dfMerge['MissingPlants']==0)&(dfMerge['MissingProduction']==0)])
#     fltPercentMatch         = round(100*(intSuccessMatch/intProduction),2)
#     
#     print('='*15+'\n'
#           +'For Year: '+str(intYear)+'\n'
#           +'Complete entries in Plants: '+str(intPlant)+'\n'
#           +'Incomplete entries in Plants: '+str(intPlantMissing)+'\n'
#           +'Complete entries in Production: '+str(intProduction)+'\n'
#           +'Incomplete entries in Production: '+str(intProductionMissing)+'\n'
#           +'Complete Plant to incomplete Production entries: '+str(intPlantToProdError)+'\n'
#           +'Complete Plant entries not matched: '+str(intPlantMismatchError)+'\n'
#           +'Complete Production to incomplete Plant entries: '+str(intProdToPlantError)+'\n'
#           +'Complete Production entries not matched: '+str(intProdMismatchError)+'\n'
#           +'Successful & Complete Matches: '+str(intSuccessMatch)+'\n'
#           +str(fltPercentMatch)+'% of original dataset'
#           )
#     
#     
#     ##  Idenitfy production variables, aggregate and transpose
#     lstNetGen = ['Netgen\nJanuary','Netgen\nFebruary','Netgen\nMarch',
#                  'Netgen\nApril','Netgen\nMay','Netgen\nJune','Netgen\nJuly',
#                  'Netgen\nAugust','Netgen\nSeptember','Netgen\nOctober',
#                  'Netgen\nNovember','Netgen\nDecember']
#     
#     #   Filled gaps, replace Nan, convert back to float, subset variable names
#     dfMerge[lstNetGen].fillna(0,inplace=True)
#     i = 0
#     while i < len(lstNetGen) :
#         dfMerge.loc[dfMerge[lstNetGen[i]]=='.',lstNetGen[i]] = 0
#         dfMerge[lstNetGen[i]] = dfMerge[lstNetGen[i]].astype(float)
#         dfMerge = dfMerge.rename(columns={lstNetGen[i]:lstNetGen[i].replace('Netgen\n','')})
#         lstNetGen[i] = lstNetGen[i].replace('Netgen\n','')
#         i+=1
#     #
#     dfMerge = aggregate_transpose(dfMerge,lstNetGen,lstAggregateBy)
#     
#     
#     
#     #df = df.rename(columns={col:'col'+str(i)})
#     dfMerge['year'] = intYear
#     dfMerge.to_csv('data/merged/'+str(lstAggregateBy[-1])+str(intYear)+'.csv',index=False)
#     intYear -= 1
# #   end of year loop
# 
# 
# 
# 
# ##  From test run
# # Complete entries in Plants: 11714
# # Incomplete entries in Plants: 119
# # Complete entries in Production: 14519
# # Incomplete entries in Production: 0
# # Complete Plant to incomplete Production entries: 0
# # Complete Plant entries not matched: 2006
# # Complete Production to incomplete Plant entries: 63
# # Complete Production entries not matched: 239
# # Successful & Complete Matches: 14222
# # 97.95% of original dataset
# #
# ##  From full 2019 Run
# # Complete entries in Plants: 11714
# # Incomplete entries in Plants: 119
# # Complete entries in Production: 14519
# # Incomplete entries in Production: 0
# # Complete Plant to incomplete Production entries: 0
# # Complete Plant entries not matched: 2006
# # Complete Production to incomplete Plant entries: 63
# # Complete Production entries not matched: 239
# # Successful & Complete Matches: 14222
# # 97.95% of original dataset
# =============================================================================
