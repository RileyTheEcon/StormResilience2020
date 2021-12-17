# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:15:31 2021

@author: RC
"""

from pandas import DataFrame, read_csv


if __name__ == '__main__' :
    print(__doc__)
    
    
    
    
    
    #   Import sample
    dfSample = read_csv('data/working/sampleStormCounties.csv')
    print('Starting rows = '+str(len(dfSample)))
    print(dfSample.dtypes)
    
    
    
    
    
    #   Drop non-coastal counties + non-storm months
    dfSample = dfSample[['year','month','county','abrState','bStorm',
                         'recovery']
                        ][dfSample['bStorm']==1]
    intSampleLen = len(dfSample)
    print('County+Storm Combos = '+str(intSampleLen))
    
    
    
    
    
    #   Count unique state/county combos
    dfSubSample = dfSample.groupby(['abrState','county']
                                   )['recovery'].count().sort_values(ascending=False).reset_index()
    
    
    
    
    
    #   Gen bFreq - most frequent storm counties
    print(dfSubSample['recovery'].value_counts(normalize=True).cumsum().shift(1,fill_value=0).reset_index())
    print('3 or more Storms = > 39 / '+str(len(dfSubSample))+' => 92nd Percentile of Storm Freqency')
    dfSubSample['bFreq'] = dfSubSample['recovery'].apply(lambda x : 1 if x 
                                                         in [5,4,3] else 0)
    
    
    
    
    #   Merge bFreq back to original sample
    dfSample = dfSample.merge(dfSubSample[['abrState','county','bFreq']],
                              how='left',on=['abrState','county'])
    assert len(dfSample)==intSampleLen
    
    
    
    
    
    #   Sum/Average recovery for P(90) Count counties
    dfResult = dfSample[dfSample['bFreq']==1].groupby(['abrState','county']
                                                      )['recovery'].mean().sort_values(ascending=False).reset_index()
    print('\n\n\nWorst Recovering, High Storm Frequency Counties')
    print(dfResult.nlargest(10,'recovery'))
    print('\n\n\nWorst Recovering, High Storm Frequency Counties')
    print(dfResult.nsmallest(10,'recovery'))
    
    
    
    
    
    #   Export results
    dfResult.to_csv('data/storm_freq_results.csv',index=None)
    
    
    
    
    
    #   Get Ratings data
    dfRatings = read_csv('data/rating_2.csv')
    
    
    
    
    #   Merge bFreq to Ratings
    dfNewSet = dfRatings.merge(dfSubSample[['abrState','county','bFreq']],
                               how='left',on=['abrState','county'])
    dfNewSet = dfNewSet.fillna(0)
    
    
    
    
    
    #   Export new set of ratings data
    dfNewSet.to_csv('data/rating+bFreq.csv',index=None)
    
    pass
#   endif