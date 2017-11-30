import numpy
import pickle
# class ReviewProcessing:
#     def __init__(self):
#         self.review_dict = {}
        
#     def load(self,review_delim_path,review_emb_path):
#         self.review_dict = pickle.load(open(review_delim_path,"rb"))
#         self.review_emb_path = review_emb_path
    
#     def get_emb(self,review_id):
#         if not self.review_dict:
#             raise Exception("load the word index and emb files")
#         if not review_id in self.review_dict:
#             return None
#         start,length = self.review_dict[review_id]
#         with open(self.review_emb_path,"r") as f:
#             f.seek(start)
#             line = f.read(length)
#             emb = line.strip().split("\t")[4]
#             emb = numpy.array([float(x) for x in emb.split()])
#         return emb

class ReviewProcessing:
    def __init__(self):
        self.review_dict = {}
        
    def load(self,review_delim_path,review_emb_path):
        self.review_dict = pickle.load(open(review_delim_path,"rb"))
        self.review_emb_path = review_emb_path
    
    def get_emb(self,review_id):
        if not self.review_dict:
            raise Exception("load the word index and emb files")
        if not review_id in self.review_dict:
            return None
        start,length = self.review_dict[review_id]
        with open(self.review_emb_path,"r") as f:
            f.seek(start)
            line = f.read(length-1)
            #print(line)
            emb = line.strip().split("\t")[4]
            emb = numpy.array([float(x) for x in emb.split()])
        return emb