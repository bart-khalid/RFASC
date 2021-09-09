import os

from .mfile import MFile

class FCentral:
    
    def __init__(self):
        self.record = {}
    
    def __getitem__(self, fpath):
        npath = os.path.normpath(fpath)
        if not npath in self.record:
            self.record[npath] = MFile.create(npath)
        return self.record[npath]
