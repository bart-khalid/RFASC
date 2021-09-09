import os
import sys

from .config import Config
from .compiler import Compiler
from .dfile import DFile
from .dcentral import DCentral
from .fcentral import FCentral
from .srcfile import SrcFile


class Maker:
    
    def __init__(self, config_path, fout=sys.stdout):
        self.config = Config.parse(config_path)
        self.fcentral = FCentral()
        self.dcentral = DCentral(self.fcentral)
        self.compiler = Compiler(self.config.compile_cmd, fout)
        self.combiner = Compiler(self.config.combine_cmd, fout)
    
    def make(self, fpath, opath):
        srcfile = self._create_src(fpath)
        obj_mfiles = srcfile.rcompile(self.compiler)
        
        if obj_mfiles:
            final_mfile = self.fcentral[opath]
            final_dfile = DFile(
                d_mfile=final_mfile,
                target=final_mfile,
                src=obj_mfiles[0],
                h_deps=obj_mfiles[1:]
            )
            if final_dfile.should_compile():
                obj_paths = [mfile.fpath for mfile in obj_mfiles]
                inp = " ".join(obj_paths)
                self.combiner.compile(inp=inp, out=opath)
                final_mfile.await_update()
    
    # === PRIVATE ===
    
    def _create_src(self, fpath):
        return SrcFile.create(
            self.fcentral, 
            self.dcentral, 
            self.config, 
            fpath
        )
