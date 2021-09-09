import os
import time

class MFile:
    
    @staticmethod
    def create(fpath):
        mfile = MFile(fpath)
        mfile.update()
        return mfile
    
    def __init__(self, fpath):
        self.fpath = fpath
        self.exists = False
        self.mtime = None
    
    def parse(self):
        with open(self.fpath, "r") as f:
            for line in f:
                yield line.rstrip()
    
    def rm(self):
        if self.exists:
            os.remove(self.fpath)
            self.exists = False
            self.mtime = None
    
    def __hash__(self):
        return hash(self.fpath)
    
    def __eq__(self, mfile):
        return self.fpath == mfile.fpath
    
    def __lt__(self, mfile):
        return self.mtime < mfile.mtime
    
    def update(self):
        self.exists = os.path.isfile(self.fpath)
        updated = False
        if self.exists:
            new_mtime = os.path.getmtime(self.fpath)
            updated = self.mtime is None or new_mtime > self.mtime
            self.mtime = new_mtime
        return updated
    
    def await_update(self, tries=600000, rest_dt=0.01):
        "Wait up to a default of 10 minutes while checking for update."
        for i in range(tries):
            if self.update():
                return
            else:
                time.sleep(rest_dt)
        raise TimeoutError("Waited %.2f s." % (tries * rest_dt))
