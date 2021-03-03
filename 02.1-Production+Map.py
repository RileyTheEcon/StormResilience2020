# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:24:59 2020
@author: RC

Input files:        data/stem.csv,
                    data/working/Production_collapsed.csv,
                    data/source/county maps/cb_2018_us_county_500k.shp,
                    data/censusMaster.csv,
Preceeding Script:  
Output files:       data/working/Production-filled.csv
Following Script:   

This takes the production results and matches it to the county-level data to 
the Census map. For counties missing production data, this matches those 
counties with the nearest county with production data. For the purposes of
tracking the matching, the summary stats of the distances are reported.

"""










# =========================================================================== #
from pandas import *
from geopandas import *
import matplotlib.pyplot as plt
from shapely.geometry import Point,LineString,shape
from shapely.ops import nearest_points
from math import *

set_option('display.max_rows',100)

replaceHurr1 = [('abrCounty','LaPorte','La Porte'),
                ('abrCounty','DeKalb','De Kalb'),
                ('abrCounty','LaGrange','La Grange'),
                ('abrCounty','DeSoto','De Soto'),
                ('abrCounty','LaSalle','La Salle'),
                ('abrCounty','LaGrange','La Grange'),
                ('abrCounty','DeWitt','De Witt'),
                ('abrCounty','DuPage','Du Page'),
                ('abrCounty','Doña Ana','Dona Ana'),
                ('abrCounty','LaMoure','La Moure')]

replaceHurr2 = [('stateCode',11,'abrCounty','District of Columbia','Washington'),
                ('stateCode',51,'abrCounty','Newport News','Newport News City'),
                ('stateCode',51,'abrCounty','Suffolk','Suffolk City'),
                ('stateCode',51,'abrCounty','Norfolk','Norfolk City'),
                ('stateCode',51,'abrCounty','Portsmouth','Portsmouth City'),
                ('stateCode',51,'abrCounty','Hampton','Hampton City'),
                ('stateCode',51,'abrCounty','Norton','Norton City'),
                ('stateCode',51,'abrCounty','Virginia Beach','Virginia Beach City'),
                ('stateCode',51,'abrCounty','Fredericksburg','Fredericksburg City'),
                ('stateCode',51,'abrCounty','Manassas Park','Manassas Park City'),
                ('stateCode',51,'abrCounty','Charlottesville','Charlottesville City'),
                ('stateCode',51,'abrCounty','Colonial Heights','Colonial Heights City'),
                ('stateCode',51,'abrCounty','Falls Church','Falls Church City'),
                ('stateCode',51,'abrCounty','Staunton','Staunton City'),
                ('stateCode',51,'abrCounty','Williamsburg','Williamsburg City'),
                ('stateCode',51,'abrCounty','Alexandria','Alexandria City'),
                ('stateCode',51,'abrCounty','Covington','Covington City'),
                ('stateCode',51,'abrCounty','Lynchburg','Lynchburg City'),
                ('stateCode',51,'abrCounty','Manassas','Manassas City'),
                ('stateCode',51,'abrCounty','Chesapeake','Chesapeake City'),
                ('stateCode',51,'abrCounty','Bristol','Bristol City'),
                ('stateCode',51,'abrCounty','Radford','Radford City'),
                ('stateCode',51,'abrCounty','Winchester','Winchester City'),
                ('stateCode',51,'abrCounty','Danville','Danville City'),
                ('stateCode',51,'abrCounty','Waynesboro','Waynesboro City'),
                ('stateCode',51,'abrCounty','Hopewell','Hopewell City'),
                ('stateCode',51,'abrCounty','Buena Vista','Buena Vista City'),
                ('stateCode',51,'abrCounty','Salem','Salem City'),
                ('stateCode',51,'abrCounty','Poquoson','Poquoson City'),
                ('stateCode',51,'abrCounty','Harrisonburg','Harrisonburg City'),
                ('stateCode',51,'abrCounty','Galax','Galax City'),
                ('stateCode',51,'abrCounty','Petersburg','Petersburg City'),
                ('stateCode',51,'abrCounty','Martinsville','Martinsville City'),
                ('stateCode',51,'abrCounty','Emporia','Emporia City'),
                ('stateCode',51,'abrCounty','Lexington','Lexington City')
               ]
# ('stateCode',,'abrCounty','',''),

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
def remove_vars (df,listVar) :
    for var in listVar :
        del df[var]
    return df
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
def rename_entries_1 (df,listReplace) :
    # Var1 , Entry, New Entry
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1]),
               replace[0]] = replace[2]
    ####
    return df
####
def rename_entries_2 (df,listReplace) :
    for replace in listReplace :
        df.loc[(df[replace[0]]==replace[1])&
               (df[replace[2]]==replace[3]),
               replace[2]] = replace[4]
    ####
    return df
####
def fix_dupe_location (df) :
    def add_city (df,listReplace) :
        for replace in listReplace :
            df.loc[(df['index']==replace[0]),
                   'abrCounty'] = replace[1]+' City'
        #   end for
        return df
    ####
    # pass shapeMap
    df = df.reset_index()
    mapDupe = df[['index','abrCounty','stateCode']][df[['abrCounty','stateCode']].duplicated()]
    listFix = []
    for index,row in mapDupe.iterrows() :
        listFix.append((row['index'],row['abrCounty'],row['stateCode']))
    #   end for
    df = add_city(df,listFix)
    del df['index']
    return df
####
def nearest_point (x,pts) :
    index = dfHasProd['near-Point']==nearest_points(x,pts)[1]
    nearest = dfHasProd['near-Index'][index].values[0]
    return nearest
####
def distance (lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    d = 2 * 3958.8 * asin(sqrt(a)) # radius of earth = 3958.8 miles
    return d
####
def get_distance (tpl) :
    lon1 = tpl[0].x
    lat1 = tpl[0].y
    lon2 = tpl[1].x
    lat2 = tpl[1].y
    
    x = distance(lat1,lon1,lat2,lon2)
    
    return x
####
# =========================================================================== #





# =========================================================================== #
if __name__ == '__main__' : 
    print(__doc__)
    #   Import dfStem
    dfStem = read_csv('data/stem.csv') #['year','month','state','county']
    dfStem = dfStem[dfStem['year']<2020]
    
    #   Import dfProduction-collapsed
    dfProd = read_csv('data/working/Production_collapsed.csv')
    dfProd['month'] = dfProd['Month'].apply(month_to_int)
    dfProd['year'] = dfProd['Year']
    dfProd = dfProd[dfProd['Year']>=2010]
    dfProd = remove_vars(dfProd,['Month','Year'])
    
    #   Merge production onto stem w [state,county]
    dfStem = merge_track(dfStem,dfProd,['state','county','year','month'],'left')[0]
    dfStem = remove_vars(dfStem,['fromMerger','MergeSuccess','fromSource'])
    
    #   Gen bProduction
    dfStem['bProduction'] = dfStem['Production'].apply(lambda x : 1 if notnull(x) else 0)
    
    
    
    
    
    
    
    
    
    
    #   Import shapefile
    shapeMap = read_file('data/source/county maps/cb_2018_us_county_500k.shp')
    shapeMap = shapeMap[['STATEFP','NAME','geometry']]
    
    #   Rename STATEFP to stateCode
    shapeMap.rename(columns={'STATEFP':'stateCode',
                             'NAME':'abrCounty'},inplace=True)
    shapeMap['stateCode'] = to_numeric(shapeMap['stateCode'],errors='coerce')
    
    #   Add city to cities
    print(shapeMap['abrCounty'][shapeMap[['abrCounty','stateCode']].duplicated(keep=False)])
    shapeMap = fix_dupe_location(shapeMap)
    print(shapeMap['abrCounty'][shapeMap[['abrCounty','stateCode']].duplicated(keep=False)])
    
    #   Remove territories
    shapeMap = shapeMap[(shapeMap['stateCode']!=78)&    # Virgin Islands
                        (shapeMap['stateCode']!=72)&    # Puerto Rico
                        (shapeMap['stateCode']!=69)&    # Mariana Islands
                        (shapeMap['stateCode']!=66)&    # Guam
                        (shapeMap['stateCode']!=60)     # American Samoa
                        ] 
    
    #   Rename abrCounty
    shapeMap = rename_entries_1(shapeMap,replaceHurr1)
    shapeMap = rename_entries_2(shapeMap,replaceHurr2)
    
    #   Add linking cols from censusMaster to shapeMap
    # shapeMap has stateCode + abrCounty ; needs state + county
    print(len(shapeMap))
    dfCensus = read_csv('data/censusMaster.csv')
    dfCounty = dfCensus[['stateCode','abrCounty','state','county']].drop_duplicates()
    shapeMap = merge_track(shapeMap,dfCounty,['stateCode','abrCounty'],'left')[0]
    shapeMap = remove_vars(shapeMap,['fromMerger','fromSource','MergeSuccess'])
    print(len(shapeMap))
    
    
    
    
    
    
    
    
    
    
    #   Merge shapemap onto stem w [stateCode,abrCounty]
    dfStem = merge_track(dfStem,shapeMap,['state','county'],'left')[0]
    
    #   Add Label Point
    dfStem['point'] = dfStem['geometry'].apply(lambda x : x.centroid.coords[:])
    listPoints = [Point(coords[0]) for coords in dfStem['point']]
    dfStem = dfStem[['year','month','state','county','Production','bProduction']]
    dfStem = GeoDataFrame(dfStem,geometry=listPoints)
    dfStem = dfStem.reset_index()
    
    
    
    
    
    
    
    
    
    
    #   Create Separate Dfs
    # Months+Counties w  Production: 226920/377040 = 60.185%
    # Months+Counties wo Production: 150120/377040 = 39.815%
    listOut = []
    intMonthCounter = 1
    dfTracker = DataFrame()
    for year in [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019] :
        for month in [1,2,3,4,5,6,7,8,9,10,11,12] :
            # Take temporal snapshot
            df = dfStem[(dfStem['year']==year)&(dfStem['month']==month)]
            
            # Grab non-prod counties, remove empty prod col
            dfNoProd  = df[df['bProduction']==0]
            dfNoProd = remove_vars(dfNoProd,['Production'])
            
            # Grab prod counties, combine geos, rename cols to avoid
            # overwrites on merge, set distance=0
            dfHasProd = df[df['bProduction']==1]
            ptsProd = dfHasProd['geometry'].unary_union     # needs geo col preserved before rename
            dfHasProd.rename(columns={'index':'near-Index',
                                      'geometry':'near-Point'},inplace=True)
            dfHasProd['distance'] = 0
            
            #   Find index of closed point
            dfNoProd['near-Index'] = dfNoProd['geometry'].apply(nearest_point,args=(ptsProd,))
            
            #   Merge production onto missing-production df
            dfNoProd = merge_track(dfNoProd,dfHasProd[['near-Index','near-Point','Production']],
                                   ['near-Index'],'left')[0]
    
            #   Calculate distance
            dfNoProd['distance'] = dfNoProd[['geometry','near-Point']].apply(get_distance,axis=1)
            
            #   Drop unnecessary variables
            dfHasProd = remove_vars(dfHasProd,['near-Index','near-Point'])
            dfNoProd = remove_vars(dfNoProd,['index','geometry','near-Index',
                                             'near-Point','fromMerger',
                                             'fromSource','MergeSuccess'])
            
            #   Add dfs to output list
            listOut = listOut+[dfHasProd,dfNoProd]
            
            #   Result tracker
            dictRow = {}
            dictRow['month'] = intMonthCounter
            dictRow['prop'] = 100*(len(dfNoProd)/len(df))
            dictRow['min'] = dfNoProd['distance'].min()
            dictRow['p25'] = dfNoProd['distance'].quantile(0.25)
            dictRow['medi'] = dfNoProd['distance'].median()
            dictRow['mean'] = dfNoProd['distance'].mean()
            dictRow['p75'] = dfNoProd['distance'].quantile(0.75)
            dictRow['max'] = dfNoProd['distance'].max()
            dfTracker = dfTracker.append(dictRow,ignore_index=True)
            intMonthCounter += 1
            print('\nDistance result for '+str(month)+'/'+str(year))
            print(dictRow)
            
        #   end for
    #   end for
    
    
    
    
    
    
    
    
    
    
    #   Combine back together
    dfOut = concat(listOut)
    dfOut = dfOut.sort_values(by=['state','county','year','month']).reset_index(drop=True)
    dfOut = dfOut[['year','month','county','state','Production','bProduction','distance']]
    dfOut.to_csv('data/working/Production-filled.csv',index=None)
    
    #   Plot distances
    plt.plot(dfTracker['month'],dfTracker['prop'],label='%No Prod')
    plt.plot(dfTracker['month'],dfTracker['min'],label='min')
    plt.plot(dfTracker['month'],dfTracker['p25'],label='p(25)')
    plt.plot(dfTracker['month'],dfTracker['medi'],label='median')
    plt.plot(dfTracker['month'],dfTracker['mean'],label='mean')
    plt.plot(dfTracker['month'],dfTracker['p75'],label='p(75)')
    plt.plot(dfTracker['month'],dfTracker['max'],label='max')
    plt.title('Energy Production Distance Results')
    
#   end main()
# =========================================================================== #



# shapeMap['stateCode'][shapeMap['MergeSuccess']==0].value_counts().sort_index()
# shapeMap['abrCounty'][(shapeMap['stateCode']==1)&(shapeMap['MergeSuccess']==0)]
# dfCounty['abrCounty'][dfCounty['stateCode']==11]
# for num in [51] :
#     print('\nFor state : '+str(num))
#     print('ShapeMap :')
#     print(shapeMap['abrCounty'][(shapeMap['stateCode']==num)&(shapeMap['MergeSuccess']==0)])
#     print('Census Master :')
#     print(dfCounty[['abrCounty','state']][dfCounty['stateCode']==num])
# #   end for






















