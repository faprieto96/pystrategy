
import numpy as np
def  MaxDrawdown(r):
    # Maximum drawdown is defined as the largest drop from a peak to a bottom
    # experienced in a certain time period.
        
    #[MDD, MDDs, MDDe, MDDr] = MAXDRAWDOWN(r)

    # INPUTS:
    # r...      vector of log returns
    #
    # OUTPUTS:
    # MDD...   Maximum drawdown expressed as a log return figure
    # MDDs...  Start of maximum drawdown period expressed as an
    #          index on vector aReturnVector
    # MDDe...  End of maximum drawdown period expressed as an
    #          index on vector aReturnVector
    # MDDr...  End of recovery period expressed as an index on vector
    #          aReturnVector
    #
    # Andreas Steiner, March 2006
    # performanceanalysis@andreassteiner.net,
    # http://www.andreassteiner.net/performanceanalysis
    # size of r
    n = max(r.shape)
    # calculate vector of cum returns
    cr = np.cumsum((np.asarray(r)).flatten(), axis=0)
    # calculate drawdown vector
    dd=[]
    for i in range(1,n):
        dd.append(max(cr[0:i])-cr[i-1])
    dd=np.array(dd)
    # calculate maximum drawdown statistics
    MDD = max(dd)
    MDDe = np.where(dd==MDD)[0][0]
    try:
        MDDs = np.where(abs(cr[MDDe]+ MDD - cr) < 0.000001)[0][0]
    except:
        MDDs=0
    try:
        MDDr = np.where(MDDe+min(cr[MDDe:] >= cr[MDDs]))[0]-1
    except:
        try:
            MDDr = np.where(MDDe+min(cr>= cr[MDDs]))[0]-1
        except:
            MDDr = []
        
    return MDD, MDDs, MDDe, MDDr