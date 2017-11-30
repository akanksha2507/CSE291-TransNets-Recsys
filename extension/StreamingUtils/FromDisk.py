'''
stream from disk, records of the form: uA iB rAB UText BText 

@author: roseck
Created on Mar 11, 2017
'''

import time, gzip,numpy
from DatasetUtils import Misc
from DatasetUtils import ReviewProcessing

class FromDisk():
    
    def __init__(self, filename,review_delim,review_emb):
        '''
        filename is the dataset to iterate on
        '''
    
        if filename.endswith('.gz'):
            self.fin = gzip.open(filename, 'r')
        else:
            self.fin = open(filename, 'r',encoding="utf-8")
        self.cache = {}
        self.tot_batch = 0
        self.process = ReviewProcessing.ReviewProcessing()
        self.process.load(review_delim,review_emb)
        

    
    def _close(self):
        self.fin.close()
    
    def emb_list(self,text):
        emb = []
        for rev_id in text.split(" "):
            if rev_id in self.cache:
                    _emb = self.cache[rev_id]
            else:
               _emb = self.process.get_emb(rev_id)
               if len(self.cache) < 100000:
                    self.cache[rev_id] = _emb
            emb.append(_emb)########
        return emb

    def BatchIter(self, batch_size):    
        '''
        batch size = number of training u,b,r examples in the batch
        returns:
        uList = uA useres
        bList = iB items
        rList: rAB (float)
        user_revlist: the UText converted to int list
        item_revlist: the BText converted to int list
         
        '''
        while True:
            #one batch
            start = time.time()
            uList = []
            bList = []
            rList = []
            uTextList = []
            bTextList = []
            ubRevList = []
            
            
            for line in self.fin:
                vals = line.strip().split("\t")
                if len(vals) == 0:
                    continue
            
                try:
                   u = vals[0]
                   b = vals[1]
                   r = float(vals[2])
                   uText = vals[3]
                   bText = vals[4]
                   rev = vals[5]
                
                   uList.append(u)
                   bList.append(b)
                   rList.append(r)
                   uTextList.append(self.emb_list(uText))
                   bTextList.append(self.emb_list(bText))
                   ubRevList.append(Misc.int_list(rev))
                except :
                   continue
                
                if len(uList) >= batch_size:
                    break
            
            if len(uList) == 0:
                #end of data
                self._close()
                #print ('Total Batch gen time = ', (self.tot_batch/60.0), ' min')
                raise StopIteration
            
            end = time.time()
            
            bg = (end - start)
            
            #print ('Batch gen time = ', bg, ' sec')
            
            self.tot_batch += bg
            
            yield uList, bList, rList, uTextList, bTextList, ubRevList

 
     

    

