"""Publication boundaries preserve the model's three-sample future context."""
import numpy as np

def sections(sample_times,duration,seconds=8,context=3):
    times=np.asarray(sample_times)
    if duration<=0 or seconds<=0 or context<0 or np.any(np.diff(times)<=0):raise ValueError('Invalid incremental timeline')
    for start in np.arange(0,duration,seconds):
        end=min(start+seconds,duration)
        first,stop=map(int,np.searchsorted(times,[start,end]))
        if stop>first:yield first,stop,min(len(times),stop+context),end
