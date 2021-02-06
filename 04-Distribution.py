# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:40:15 2020
@author: RC

Input files:        data/working/takeupYearly.csv
Preceeding Script:  
Output files:       
Following Script:   

This takes the takeupYearly data (used for reporting) and uses it to create a
set of histograms and summary stats tables.

"""










# =========================================================================== #
from pandas import *
from numpy import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

bDropZe = 1
strFile = 'data/working/takeupYearly.csv'
intYear = 2018
listVar = ['protectGap','takeupTotal','InsurableValue','TotalInsurableValue']
listPer = [0.005,0.025,0.16,0.5,0.84,0.975,0.995]

# =========================================================================== #










# =========================================================================== #

# =========================================================================== #










# =========================================================================== #
if __name__ == '__main__' :
    print(__doc__)
    listPer.sort()
    dfFull = read_csv(strFile)
    dfOut = DataFrame()
    
    
    
    
    
    
    
    
    
    
    for var in listVar :
        dictRow = {}
        
        #   Take var and subset df
        dictRow['var'] = var
        df = dfFull[var][(dfFull['year']==intYear)&(dfFull[var]!=0)]
        df = log(df)
        
        
        
        
        
        
        
        
        
        
        #   Get summary stats
        dfMean = df.mean()
        dictRow['popStdD'] = std(df)
        
        
        
        
        
        
        
        
        
        
        #   Run thru percents
        for percent in listPer :
            dictRow['cdf'+str(percent)] = (norm.ppf(percent)*dictRow['popStdD'])+dfMean
        #   end for
        
        
        
        
        
        
        
        
        
        
        #   Add to output frame
        dfOut = dfOut.append(dictRow,True)
        
        
        
        
        
        
        
        
        
        
        #   Graph distributions
        rangeNorm = arange(-3.5,3.5,0.001)
        # dfFull[var].plot(kind='hist',density=True)
        # plt.plot(rangeNorm,norm.pdf(rangeNorm,0,1))
        # rangeNorm = arange(-1,4,0.001)
        plt.figure()
        plt.hist(df,bins=10)    # 25
        plt.plot(rangeNorm*dictRow['popStdD']+dictRow['cdf0.5'],
                 len(df)*norm.pdf(rangeNorm,0,1))
        plt.title('Distribution of '+var)
        plt.savefig('plot/dist/'+var+'.png')
        # sns.distplot(dfFull[var],hist=True)
     #   end for
    
    
    
    
    
    
    
    
    
    
    #   Sort output cols
    dfOut.rename(columns={'cdf0.5':'mean'},inplace=True)
    listCols = list(dfOut)
    listCols = [x for x in listCols if x not in ['var','popStdD']]
    dfOut    = dfOut[['var','popStdD']+listCols]
    
    
    
    
    
    
    
    
    
    
    dfOut.to_csv('takeup-rate-distribution.csv',index=None)
    
#   end main()
# =========================================================================== #





















