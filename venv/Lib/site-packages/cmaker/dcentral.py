from .dfile import DFile


class DCentral:
    
    def __init__(self, fcentral):
        self.fcentral = fcentral
        self.record = {}
    
    def __getitem__(self, mfile):
        if not mfile in self.record:
            self.record[mfile] = DFile.parse(self.fcentral, mfile)
        return self.record[mfile]
