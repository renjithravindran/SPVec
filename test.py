"""

"""

import unittest
import tempfile
import numpy as np

from math import log2

from cmdvec import CMDVec

class basic_tests(unittest.TestCase):
    """
    """
    
    def setUp(self):
        test = b"a a c b c a a d \n c a d b c c a c"
        self.tfile = tempfile.NamedTemporaryFile()
        self.tfile.write(test)
        self.tfile.seek(0)
        filename = self.tfile.name
        self.mod1 = CMDVec(corpus_filename=filename,windowtype='left&right',windowsize=1,jobs=2)
    def test_coocs(self):
        """
        coocs windowtype='left&right',windowsize=1 
        for "a a c b c a a d c a d b c c a c"
        are
        aa,aa,ac,ca,cb,bc,bc,cb,ca,ac,aa,aa,ad,da,
        ca,ac,ad,da,db,bd,bc,cb,cc,cc,ca,ac,ac,ca,
        """
                          # a,c,b,d   
        valid_cooc_mat =  [[4,5,0,2],# a
                           [5,2,3,0],# c
                           [0,3,0,1],# b
                           [2,0,1,0]]# d
        obtained_coo_mat = self.mod1.coocs_raw.todense().tolist()
        
        self.assertEqual(obtained_coo_mat,valid_cooc_mat,"wrong cooc computation")
    
    def test_pmi(self):
        """
        M : data matrix
        M(x,y) : cooc of x and y
        rsM(x) : row sum of x in M
        sM : sum of all values in M
        
        pmi = log2(M(x,y)) + log2(sM) - log2(rsM(x)) - log2(rsM(y))
        """
        M = self.mod1.coocs_raw.todense().tolist()
        rsM = [sum(l) for l in M]
        sM = sum(rsM)
        valid_pmi = [[0.0,0.0,0.0,0.0],
                     [0.0,0.0,0.0,0.0],
                     [0.0,0.0,0.0,0.0],
                     [0.0,0.0,0.0,0.0]]
        valid_pmi = np.array(valid_pmi,dtype=np.dtype('f4'))
        
        for i in range(len(M)):
            for j in range(len(M[0])):
                if M[i][j] > 0:
                    valid_pmi [i,j] = log2(M[i][j]) + log2(sM) - log2(rsM[i]) - log2(rsM[j])
        
        valid_pmi = np.round(valid_pmi,3)
        
        self.mod1.make_pmiMat()
        obtained_pmi = np.round(self.mod1.coocs_pmi.todense(),3)
        #valid_pmi = valid_pmi.tolist()
        #obtained_pmi =  obtained_pmi.tolist()
        
        allTrue = (obtained_pmi == valid_pmi).all()
        
        self.assertEqual(allTrue,True,"wrong pmi computation")
        
        
        
        

if __name__ == '__main__':
    unittest.main()
    