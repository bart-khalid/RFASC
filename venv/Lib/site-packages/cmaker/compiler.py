import os
import sys

ERROR = "Compilation returned non-zero status."

class Compiler:
    
    class Error(Exception):
        
        def __init__(self):
            super().__init__(ERROR)
    
    def __init__(self, cmd, fout=sys.stdout):
        self.cmd = cmd
        self.fout = fout
    
    def compile(self, inp, out):
        dpath = os.path.dirname(out)
        if dpath and not os.path.isdir(dpath):
            os.makedirs(dpath)
        
        cmd = self.cmd.format(inp=inp, out=out)
        self.fout.write(cmd + "\n")
        self.fout.flush()
        
        if os.system(cmd):
            raise Compiler.Error()
