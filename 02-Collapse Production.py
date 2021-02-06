# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:22:36 2020
@author: RC

Input files:        data/working/Production.csv,
                    data/censusMaster.csv,
Preceeding Script:  
Output files:       data/working/Production_collapsed.csv
Following Script:   

This takes the fuel type-level code from the Production code, and collapes 
those rows to the county-level.

"""










# =============================================================================
from pandas import *

set_option('display.max_rows', 100)

listGroup   = ['Month','Year','abrState','abrCounty','state','county']
listMean    = ['Latitude','Longitude']
listSum     = ['Production']

replaceProd = [('abrState','AK','abrCounty','Lake And Peninsula',       'Lake and Peninsula'),
               ('abrState','AK','abrCounty','Lake & Peninsula Bor',     'Lake and Peninsula'),
               ('abrState','AK','abrCounty','Matanuska Susitna',        'Matanuska-Susitna'),
               ('abrState','AK','abrCounty','Prince Of Wales',          'Prince of Wales-Hyder'),
               ('abrState','AK','abrCounty','Prince Of Wales Ketchikan','Prince of Wales-Hyder'),
               ('abrState','AK','abrCounty','Skagway Hoonah Angoon',    'Skagway'),
               ('abrState','AK','abrCounty','Valdez Cordova',           'Valdez-Cordova'),
               ('abrState','AK','abrCounty','Wade Hampton',             'Kusilvak'),
               ('abrState','AK','abrCounty','Wrangell Petersburg',      'Petersburg'),
               ('abrState','AK','abrCounty','Yukon Koyukuk',            'Yukon-Koyukuk'),                              
               
               ('abrState','DC','abrCounty','District Of Columbia',     'Washington'),
               ('abrState','FL','abrCounty','Miami Dade',               'Miami-Dade'),               
               
               ('abrState','IL','abrCounty','Dupage',                   'Du Page'),
               ('abrState','IL','abrCounty','Saint Clair',              'St. Clair'),
               ('abrState','IN','abrCounty','Laporte',                  'La Porte'),
               
               ('abrState','MA','abrCounty','North Essex',              'Essex'),
               ('abrState','MA','abrCounty','Somerset',                 'Bristol'),
               ('abrState','MD','abrCounty','Prince Georges',           "Prince George's"),
               ('abrState','MD','abrCounty','Queen Annes',              "Queen Anne's"),
               ('abrState','MN','abrCounty','Osceola',                  'Renville'),
               ('abrState','MS','abrCounty','Clark',                    'Clarke'),
               ('abrState','MT','abrCounty','Lewis And Clark',          'Lewis and Clark'),
               
               ('abrState','NC','abrCounty','Hanover',                  'New Hanover'),
               ('abrState','ND','abrCounty','Lamoure',                  'La Moure'),
               
               ('abrState','NH','abrCounty','New Hampshire',            'Coos'),
               ('abrState','NH','abrCounty','Plaquemines',              'Coos'),
               
               ('abrState','PA','abrCounty','Mongomery',                'Montgomery'),               
               
               ('abrState','VA','abrCounty','Alexandria',               'Alexandria City'),
               ('abrState','VA','abrCounty','Chesapeake',               'Chesapeake City'),
               ('abrState','VA','abrCounty','City Of Hopewell',         'Hopewell City'),
               ('abrState','VA','abrCounty','Danville',                 'Danville City'),
               ('abrState','VA','abrCounty','Hampton',                  'Hampton City'),
               ('abrState','VA','abrCounty','Hopewell',                 'Hopewell City'),
               ('abrState','VA','abrCounty','Isle Of Wight',            'Isle of Wight'),
               ('abrState','VA','abrCounty','King And Queen',           'King and Queen'),
               
               ('abrState','VT','abrCounty','Eindham',                  'Windham'),
               
               ('abrState','WI','abrCounty','Fond Du Lac',              'Fond du Lac')
               ]
# =============================================================================










# =============================================================================
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
####
def print_unique(df,listVars) :
    df = df[listVars].drop_duplicates(subset=listVars)
    df = df.sort_values(by=listVars)
    print(df)
####
def fix_basics (x) :
    if x.find('St ')==0 :
        x = x.replace('St ','St. ')
    ####
    elif x.find('Mc')==0 :
        x = 'Mc'+x[2:].title()
    ####
    elif x.find('O ')==0:
        x = 'O\''+x[2:]
    elif x=='Dekalb' : x = 'De Kalb'
    elif x=='Desoto' : x = 'De Soto'
    elif x=='Dewitt' : x = 'De Witt'
    elif x=='Lasalle' : x = 'La Salle'
    return x
####
def rename_entries (df,listReplace) :
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1])&
               (df[replace[2]]==replace[3]),
               replace[2]] = replace[4]
    ####
    return df
####
# =============================================================================










# =============================================================================
if __name__ == '__main__' :
    print(__doc__)
    dfProduction = read_csv('data/working/Production.csv')
    dfCensus = read_csv('data/censusMaster.csv')



    
    
    
    
    
    
    
    #   Get unique county list
    # Get unique
    dfCensus = dfCensus.drop_duplicates(subset=['county','state'])
    dfCensus = dfCensus[['state','county','abrCounty','abrState']]
    
    
    
        
    
    
    
    
    
    
    #   Production data
    # Rename State and County -> abrState and abrCounty
    dfProduction.rename(columns={'State':'abrState','County':'abrCounty'},
                  inplace=True)
    
    # Fix county names
    dfProduction['abrCounty'] = dfProduction['abrCounty'].str.title()
    dfProduction['abrCounty'] = dfProduction['abrCounty'].apply(fix_basics)
    dfProduction = rename_entries(dfProduction,replaceProd)
    
    dfProduction = merge_track(dfProduction,dfCensus,['abrState','abrCounty'],'left')[0]
    print_unique(dfProduction[dfProduction['MergeSuccess']==0],['abrState','abrCounty'])
    
    
    
    
    
    
    
    
    
    
    #   Gen base list
    dfProdCol = dfProduction[listGroup].drop_duplicates()
    
    #   Take sum
    df = dfProduction.groupby(listGroup)[listSum].sum().reset_index()
    dfProdCol = merge_track(dfProdCol,df,listGroup,'left')[0]
    dfProdCol = dfProdCol.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    
    #   Take mean
    df = dfProduction.groupby(listGroup)[listMean].mean().reset_index()
    dfProdCol = merge_track(dfProdCol,df,listGroup,'left')[0]
    dfProdCol = dfProdCol.drop(columns=['fromMerger','MergeSuccess','fromSource'])
    
    
    
    
    
    
    
    
    
    
    dfProdCol.to_csv('data/working/Production_collapsed.csv',index=None)
#   endmain
# =============================================================================

'''
'Plant Name',       = drop
'Plant Code',       = drop
'Utility Name',     = drop
'Utility ID',       = drop
'Fuel Code',        = concat
'Street Address',   = drop
'City',             = drop
'State',            = group
'Zip',              = drop
'County',           = group
'Latitude',         = mean
'Longitude',        = mean
'AsOf',             = drop
'MergeSuccess',     = drop
'Month',            = group
'Production',       = sum
'Year'              = group
'''
