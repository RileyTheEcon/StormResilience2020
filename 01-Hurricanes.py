# -*- coding: utf-8 -*-
"""
@author: RC

Input files:        data/source/county maps/cb_2018_us_county_500k.shp
                    tropycal.tracks
Preceeding Script:  00-Fip Master.py
Output files:       data/working/NOAA/hurrican-name_hurricane-year.csv
Following Script:   

Must be run with geopandas package in environment. 

Reads shape file from Census, gets hurricane paths from HURDAT2, tracks 
overlaps of paths and maps. Creates maps with paths. Note that most storms do
not make landfall. HURDAT2 has not been updated for 2020 storm, as last run.

    Special note: There are a dozen duplicated county-names in the Census data.
    This is the result of some states considering city limits separate from 
    county limits. This is not a major problem, since most cities are within 
    the counties named after them; however, that is not universal.
    See Richmond, VA and Franklin, VA.

"""
# =============================================================================
from pandas import *
from geopandas import *
import tropycal.tracks as tracks
import matplotlib.pyplot as plt
from shapely.geometry import Point,LineString,shape
# =============================================================================










# =============================================================================
binGenMap = 1
listMapX = [-100,-65]
listMapY = [24,50]
intMapScale = 100
#listHurr = [('irma',2017),('michael',2018),('fernand',2019)]
listHurr = [('alex',2010),
            ('two',2010),
            ('bonnie',2010),
            ('colin',2010),
            ('five',2010),
            ('danielle',2010),
            ('earl',2010),
            ('fiona',2010),
            ('gaston',2010),
            ('hermine',2010),
            ('igor',2010),
            ('julia',2010),
            ('karl',2010),
            ('lisa',2010),
            ('matthew',2010),
            ('nicole',2010),
            ('otto',2010),
            ('paula',2010),
            ('richard',2010),
            ('shary',2010),
            ('tomas',2010),
            
            ('arlene',2011),
            ('bret',2011),
            ('cindy',2011),
            ('don',2011),
            ('emily',2011),
            ('franklin',2011),
            ('gert',2011),
            ('harvey',2011),
            ('irene',2011),
            ('ten',2011),
            ('jose',2011),
            ('katia',2011),
            ('unnamed',2011),
            ('lee',2011),
            ('maria',2011),
            ('nate',2011),
            ('ophelia',2011),
            ('philippe',2011),
            ('rina',2011),
            ('sean',2011),
            
            ('alberto',2012),
            ('beryl',2012),
            ('chris',2012),
            ('debby',2012),
            ('ernesto',2012),
            ('florence',2012),
            ('helene',2012),
            ('gordon',2012),
            ('isaac',2012),
            ('joyce',2012),
            ('kirk',2012),
            ('leslie',2012),
            ('michael',2012),
            ('nadine',2012),
            ('oscar',2012),
            ('patty',2012),
            ('rafael',2012),
            ('sandy',2012),
            ('tony',2012),
            
            ('andrea',2013),
            ('barry',2013),
            ('chantal',2013),
            ('dorian',2013),
            ('erin',2013),
            ('fernand',2013),
            ('gabrielle',2013),
            ('eight',2013),
            ('humberto',2013),
            ('ingrid',2013),
            ('jerry',2013),
            ('karen',2013),
            ('lorenzo',2013),
            ('melissa',2013),
            ('unnamed',2013),
            
            ('arthur',2014),
            ('two',2014),
            ('bertha',2014),
            ('cristobal',2014),
            ('dolly',2014),
            ('edouard',2014),
            ('fay',2014),
            ('gonzalo',2014),
            ('hanna',2014),
            
            ('ana',2015),
            ('bill',2015),
            ('claudette',2015),
            ('danny',2015),
            ('erika',2015),
            ('fred',2015),
            ('grace',2015),
            ('henri',2015),
            ('nine',2015),
            ('ida',2015),
            ('joaquin',2015),
            ('kate',2015),
            
            ('alex',2016),
            ('bonnie',2016),
            ('colin',2016),
            ('danielle',2016),
            ('earl',2016),
            ('fiona',2016),
            ('gaston',2016),
            ('eight',2016),
            ('hermine',2016),
            ('ian',2016),
            ('julia',2016),
            ('karl',2016),
            ('lisa',2016),
            ('matthew',2016),
            ('nicole',2016),
            ('otto',2016),
            
            ('arlene',2017),
            ('bret',2017),
            ('cindy',2017),
            ('four',2017),
            ('don',2017),
            ('emily',2017),
            ('franklin',2017),
            ('gert',2017),
            ('harvey',2017),
            ('irma',2017),
            ('jose',2017),
            ('katia',2017),
            ('lee',2017),
            ('maria',2017),
            ('nate',2017),
            ('ophelia',2017),
            ('philippe',2017),
            ('rina',2017),
            
            ('alberto',2018),
            ('beryl',2018),
            ('chris',2018),
            ('debby',2018),
            ('ernesto',2018),
            ('florence',2018),
            ('gordon',2018),
            ('helene',2018),
            ('isaac',2018),
            ('joyce',2018),
            ('eleven',2018),
            ('kirk',2018),
            ('leslie',2018),
            ('michael',2018),
            ('nadine',2018),
            ('oscar',2018),
            
            ('andrea',2019),
            ('barry',2019),
            ('three',2019),
            ('chantal',2019),
            ('dorian',2019),
            ('erin',2019),
            ('fernand',2019),
            ('gabrielle',2019),
            ('humberto',2019),
            ('jerry',2019),
            ('imelda',2019),
            ('karen',2019),
            ('lorenzo',2019),
            ('melissa',2019),
            ('fifteen',2019),
            ('nestor',2019),
            ('olga',2019),
            ('pablo',2019),
            ('rebekah',2019),
            ('sebastien',2019)
            ]       # hurricanes = 166
# =============================================================================










# =============================================================================
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
def add_county_state (df) :
    dfCensus = read_csv('data/censusMaster.csv')
    dfCensus = dfCensus[['abrState','stateCode']].drop_duplicates(subset=['abrState'],keep='first')
    dfCensus.rename(columns={'stateCode':'STATEFP'},inplace=True)
    
    df['STATEFP'] = to_numeric(df['STATEFP'],errors='coerce')
    
    df = merge_track(df,dfCensus,'STATEFP','left')[0]
    
    df['location'] = df['NAME']+', '+df['abrState']
    
    return df['location']
####
def gen_storm_df (duoHurr) :
    dfStorm = basinNA.get_storm(duoHurr).to_dataframe()
    #dfStorm['CAT'] = dfStorm[['type','vmax']].apply(storm_cat)
    
    dfStorm['year'] = dfStorm['date'].apply(isolate_better,args=('','-'))
    dfStorm['year'] = to_numeric(dfStorm['year'],errors='coerce')
    
    dfStorm['month'] = dfStorm['date'].apply(isolate_better,args=('-','-'))
    dfStorm['month'] = to_numeric(dfStorm['month'],errors='coerce')

    dfStorm['hour'] = dfStorm['date'].apply(isolate_better,args=(' ',':'))
    dfStorm['hour'] = to_numeric(dfStorm['hour'],errors='coerce')
    
    return dfStorm
####
def storm_cat (row) :
    # Assumes row: ['type','vmax']
    # TD = Tropical Depression
    # TS = Tropical Storm
    # HU = Hurricane
    # EX = Extratropical Cyclone
    # SD = Subtropical Depression
    # SS = Subtropical Storm
    # LO = A low that is not TS, EX or SS
    # WV = Tropical Wave
    # DB = Disturbance
    if row[0]=='HU' :
        x = row[1]
        if   64<=x<=82      : cat = 'H1'
        elif 83<=x<=95      : cat = 'H2'
        elif 96<=x<=112     : cat = 'H3'
        elif 113<=x<=136    : cat = 'H4'
        elif 137<=x         : cat = 'H5'
        else : cat = 's'
    else :
        cat = row[0]
    ####
    return cat
####
def return_prefer (listInput) :
    if 'HU' in listInput : x = 'HU'
    elif 'TS' in listInput : x = 'TS'
    elif 'TD' in listInput : x = 'TD'
    elif 'EX' in listInput : x = 'EX'
    elif 'SD' in listInput : x = 'SD'
    elif 'SS' in listInput : x = 'SS'
    elif 'LO' in listInput : x = 'LO'
    elif 'WV' in listInput : x = 'WV'
    elif 'DB' in listInput : x = 'DB'
    else : x = '??'

    return x
####
def long_lat_to_line (df) :
    #   Requires 'lon', 'lat' columns in df
    #   Returns LineShape Object
    listLonLat = [Point(xy) for xy in zip(df['lon'],df['lat'])]
    dfGeo = GeoDataFrame(geometry=listLonLat)
    shapeLine = LineString(dfGeo['geometry'])
    return shapeLine
####
def get_eye_paths (dfStorm) :
    strLastRow = ''
    dfGeoLines = GeoDataFrame()
    for index,row in dfStorm.iterrows() :
        if strLastRow=='' :
            dfCAT = DataFrame()
            strLastRow = row['CAT']
        ####
        dictNewRow = {'lat':row['lat'],'lon':row['lon'],'CAT':row['CAT']}
        dfCAT = dfCAT.append(dictNewRow,ignore_index=True)
        if strLastRow!=row['CAT'] :
            #   Get shape
            print(dfCAT)
            print('\n')
            shapeLine = long_lat_to_line(dfCAT)
            
            #   Add shape + CAT to geo frame
            dictGeoRow = {'geometry':shapeLine,'CAT':strLastRow}
            dfGeoLines = dfGeoLines.append(dictGeoRow,ignore_index=True)
            
            #   Start new subset + def new last row
            dfCAT = DataFrame()
            dfCAT = dfCAT.append(dictNewRow,ignore_index=True)
            strLastRow = row['CAT']
        if index==(len(dfStorm)-1) :
            #   Get shape
            print(dfCAT)
            print('\n')
            shapeLine = long_lat_to_line(dfCAT)
            
            #   Add shape + CAT to geo frame
            dictGeoRow = {'geometry':shapeLine,'CAT':strLastRow}
            dfGeoLines = dfGeoLines.append(dictGeoRow,ignore_index=True)
        ####
    ####
    return dfGeoLines
####
def storm_to_points (df) :
    listGeometry = [Point(xy) for xy in zip(df['lon'],df['lat'])]
    dfGeo = GeoDataFrame(df,geometry=listGeometry)
    return dfGeo
####
def points_to_line (df) :
    shapeLine = LineString(df['geometry'])
    return shapeLine
####
def hour_to_time (start,end) :
    if end==0 : x = 24 - start
    elif end-start<0 : x = end - start + 24
    else : x = end - start
    return x
####
def get_eye_path2 (dfGeo) :
    # Expected vars: ['year','month','type','vmax','mslp','geometry']
    dfSubset = dfGeo
    df = GeoDataFrame()
    df2 = GeoDataFrame()
    for index,row in dfSubset.iterrows() :
        df = df.append(dict(row),ignore_index=True)
        if index>0 :
            #   Start New Output Row Dict
            dictNewRow = {}
            
            #   Get non-Mean Items
            dictNewRow['year']      = row['year']
            dictNewRow['month']     = row['month']
            dictNewRow['type']      = row['type']

            #   Add row to working DF
            df = df.append(dict(row),ignore_index=True)

            #   Get Means of Things w Means
            dictNewRow['vmax']      = round(df['vmax'].max())
            dictNewRow['mslp']      = round(df['mslp'].mean())
            dictNewRow['hour']      = hour_to_time(list(df['hour'])[0],
                                                   list(df['hour'])[1])
            
            #   Get Line Shapes
            dictNewRow['geometry']  = points_to_line(df)
            
            #   Append to Output DF
            df2 = df2.append(dictNewRow,ignore_index=True)
            
            #   Reset Working DF for next iteration
            df = GeoDataFrame()
            df = df.append(dict(row),ignore_index=True)
        ####
    ####
    return df2
####
def get_time_estimate (df) :
    df['segmentLength'] = df['geometry'].apply(lambda x : x.length)
    df['time'] = df['hour'] * (df['segmentLength']/df['fullLength'])
    
    return df
####
def collapse_line_rows (dfInput) :        
    dfSingle = dfInput[~dfInput['location'].duplicated(keep=False)]
    dfDupes = dfInput[dfInput['location'].duplicated(keep=False)]
    listDupes = list(dfDupes['location'].unique())
    dfNew = DataFrame()
    for county in listDupes :
        dictNewRow = {}
        dfSubset = dfDupes[dfDupes['location']==county]
        dictNewRow['location'] = county
        dictNewRow['year'] = round(dfSubset['year'].mean())
        dictNewRow['month'] = round(dfSubset['month'].mean())
        dictNewRow['vmax'] = dfSubset['vmax'].max()
        dictNewRow['mslp'] = dfSubset['mslp'].mean()
        dictNewRow['time'] = dfSubset['time'].sum()
        dictNewRow['type'] = return_prefer(list(dfSubset['type'].unique()))
        dfNew = dfNew.append(dictNewRow,ignore_index=True)
    ####
    dfNew = dfSingle.append(dfNew,ignore_index=True)
    dfNew = dfNew[['location','year','month','type','vmax','mslp','time']]
    return dfNew
####
def collapse_point_rows (dfInput) :        
    dfSingle = dfInput[~dfInput['location'].duplicated(keep=False)]
    dfDupes = dfInput[dfInput['location'].duplicated(keep=False)]
    listDupes = list(dfDupes['location'].unique())
    dfNew = DataFrame()
    for county in listDupes :
        dictNewRow = {}
        dfSubset = dfDupes[dfDupes['location']==county]
        dictNewRow['location'] = county
        dictNewRow['year'] = round(dfSubset['year'].mean())
        dictNewRow['month'] = round(dfSubset['month'].mean())
        dictNewRow['vmax'] = dfSubset['vmax'].max()
        dictNewRow['mslp'] = dfSubset['mslp'].mean()
        dictNewRow['type'] = return_prefer(list(dfSubset['type'].unique()))
        dfNew = dfNew.append(dictNewRow,ignore_index=True)
    ####
    dfNew = dfSingle.append(dfNew,ignore_index=True)
    dfNew = dfNew[['location','year','month','type','vmax','mslp']]
    return dfNew
####
def dfs_to_unique(dfPointsInter,dfLinesInter) :
    listCounties = list(set(list(dfLinesInter['location'])
                            +list(dfPointsInter['location'])))
    dfOutput = DataFrame()
    for county in listCounties :
        if len(dfPointsInter[dfPointsInter['location']==county])>0 :
            dfSubset = dfPointsInter[dfPointsInter['location']==county]
            dfSubset['time'] = dfLinesInter['time'][dfLinesInter['location']==county].sum()
        ###
        else :
            dfSubset = dfLinesInter[dfLinesInter['location']==county]
        ####
        dfOutput = dfOutput.append(dfSubset,ignore_index=True)
    ####
    return dfOutput
####
def fix_dupe_location (df) :
    def add_city (df,listReplace) :
        for replace in listReplace :
            df.loc[(df['location']==replace[0])&
                   (df['index']==replace[1]),
                   'location'] = replace[0].replace(',',' City,')
        #   end for
        return df
    ####
    
    # pass shapeMap
    df = df.reset_index()
    mapDupe = df[df['location'].duplicated(keep=False)]
    listDupe = list(mapDupe['location'].unique())
    listFix = []
    for county in listDupe :
        listFix.append((county,mapDupe['index'][mapDupe['location']==county].max()))
    #   end for
    df = add_city(df,listFix)
    del df['index']
    return df
####
# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' : 
    print(__doc__)
    #   Import Census Map Data
    shapeMap = read_file('data/source/county maps/cb_2018_us_county_500k.shp')
    
    
    
    
    
    
    
    
    
    
    #   Add State Names + 'County, ST'
    shapeMap['location'] = add_county_state(shapeMap[['STATEFP','NAME']])
    shapeMap = shapeMap[['location','geometry']][shapeMap['location'].notnull()]
    shapeMap = fix_dupe_location(shapeMap)
    
    
    
    
    
    
    
    
    
    
    #   Run Set-Up for Mapping
    if binGenMap==1 :
        #   Get Labels for Counties
        shapeMap['labelPoint'] = shapeMap['geometry'].apply(lambda x : x.representative_point().coords[:])
        shapeMap['labelPoint'] = [coords[0] for coords in shapeMap['labelPoint']]
        #   Get Map Bounds
        fltX = intMapScale*(listMapX[1]-listMapX[0])/(listMapY[1]-listMapY[0])
        fltY = intMapScale*(listMapY[1]-listMapY[0])/(listMapX[1]-listMapX[0])
    #   endif
    
    
    
    
    
    
    
    
    
    
    #   Get Basin
    basinNA = tracks.TrackDataset(basin='north_atlantic')
    
    #   Get Log String
    strLog = '\nHurricanes without US Landfall:\n'
    
    
    
    
    
    
    
    
    
    
    #   Start Hurricane List Read
    for hurricane in listHurr :
        #   Get Hurricane+CAT DF
        dfStorm = gen_storm_df(hurricane)
        dfStorm = dfStorm[['year','month','hour','type',
                           'lat','lon','vmax','mslp']]
        
        
        
        
        
        
        
        
        
        
        #   Get Storm Points
        dfPoints = storm_to_points(dfStorm)
        dfPoints = dfPoints[['year','month','hour','type',
                                 'vmax','mslp','geometry']]
        
        
        
        
        
        
        
        
        
        
        #   Get Storm Lines
        dfLines = get_eye_path2(dfPoints)
        dfLines['fullLength'] = dfLines['geometry'].apply(lambda x : x.length)
        dfLines = dfLines[['year','month','hour','type',
                                 'vmax','mslp','fullLength','geometry']]
        
        #   Gen dfGeoLines
        dfGeoLines = dfPoints.append(dfLines,ignore_index=True)
        
        
        
        
        
        
        
        
        
        
        #   If Storm Did Not Hit the US
        if len(sjoin(shapeMap,dfGeoLines,op='intersects'))==0 :
            strLog = strLog + str(hurricane[0])+', '+str(hurricane[1])+'\n'
        #   endif
        
        #   If Storm Did Hit the US
        else :
            #   Get Lines Intersection
            # dfLinesInter = sjoin(shapeMap,dfLines,op='intersects')
            dfLinesInter = overlay(dfLines,shapeMap,how='union')
            dfLinesInter = dfLinesInter[dfLinesInter['location'].notnull()]
            
            #   Get Lines Timing
            dfLinesInter = get_time_estimate(dfLinesInter)
            
            #   Collapse to Single
            dfLinesInter = collapse_line_rows(dfLinesInter)
            
            #   Get Point Intersections
            dfPointsInter = sjoin(shapeMap,dfPoints,op='intersects')
            
            #   Collapse Points to Single
            dfPointsInter = collapse_point_rows(dfPointsInter)
            
            #   Combine Points and Lines to Unique List
            dfIntersect = dfs_to_unique(dfPointsInter,dfLinesInter)
            
            #   Get Hurricane Category
            dfIntersect['CAT'] = dfIntersect[['type','vmax']].apply(storm_cat,axis=1)
            dfIntersect = dfIntersect.merge(shapeMap[['location','labelPoint']],how='left',on='location')
            
            
            
            
            
            
            
            
            
            
            #   Run Mapping
            if binGenMap==1 :        
                #   Generate Map
                mapBase = shapeMap.plot(color='white',edgecolor='blue',
                                        figsize=(fltX,fltY))
                plt.xlim(listMapX)
                plt.ylim(listMapY)
                mapBase.set_facecolor('xkcd:grey')
                dfGeoLines.plot(ax=mapBase,color='red')
                for idx,row in dfIntersect.iterrows() :
                    plt.annotate(text=row['location'],xy=row['labelPoint'],
                                 horizontalalignment='center')
                #   endfor
                plt.savefig('plot/'+str(hurricane[1])+str(hurricane[0])+'.png',
                            bbox_inches='tight')
            #   endif
            
            
            
            
            
            
            
            
            
            
            #   Export csv
            dfIntersect.to_csv('data/working/NOAA/'+str(hurricane[1])
                               +str(hurricane[0])+'.csv',index=None)
        #   endif
    #   endfor
    
    
    
    
    
    
    
    
    
    
    print(strLog)
#   endif
# =============================================================================























