# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:53:45 2020
@author: RC

Input files:        data/working/NOAA/_hurricane_name_hurricane_year.csv
Preceeding Script:  
Output files:       data/working/hurricanePath.csv
Following Script:   


This takes the CSVs for individual storms (that made landfall) and combines
them into a single file. Note that this iterates through a list of files that
is generated from the file directory.

"""










# =========================================================================== #
from pandas import *
from os import listdir
from os.path import isfile, join
pathData = 'data/working/NOAA/'
# =========================================================================== #










# =========================================================================== #

# =========================================================================== #










# =========================================================================== #
if __main__ == '__name__' :
    print(__doc__)
    dfNOAA = DataFrame()
    
    listFiles = [f for f in listdir(pathData) if isfile(join(pathData,f))]
    
    for data in listFiles :
        df = read_csv(pathData+data)
        
        df['name'] = data[4:-4]
        
        dfNOAA = dfNOAA.append(df,ignore_index=True)
    #   endfor
    fNOAA.to_csv('data/working/hurricanePath.csv',index=None)
#   endmain
# =========================================================================== #

# dfNOAA[['location','name','time','year']].nlargest(10,'time')
# dfNOAA[['location','name','time','year']].nsmallest(10,'time')