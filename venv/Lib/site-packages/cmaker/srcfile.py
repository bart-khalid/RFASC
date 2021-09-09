import os


H_EXT = ".h"
O_EXT = ".o"
D_EXT = ".d"


class SrcFile:
    
    @staticmethod
    def create(fcentral, dcentral, config, fpath):
        base = os.path.splitext(fpath)[0]
        h_mfile = fcentral[base + H_EXT]
        src_mfile = fcentral[base + config.c_ext]
        obj_correct = base.replace(".", "_")
        obj_name = obj_correct.replace(
            "/", config.obj_dirsep
        ).replace("\\", config.obj_dirsep)
        obj_base = os.path.join(config.obj_dir, obj_name)
        obj_mfile = fcentral[obj_base + O_EXT]
        dep_mfile = fcentral[obj_base + D_EXT]
        
        dfile = None
        if dep_mfile.exists:
            dfile = dcentral[dep_mfile]
        
        return SrcFile(
            fcentral=fcentral,
            dcentral=dcentral,
            config=config,
            h_mfile=h_mfile,
            c_mfile=src_mfile,
            o_mfile=obj_mfile,
            d_mfile=dep_mfile,
            dfile=dfile
        )
    
    def rcompile(self, compiler):
        return self._rcompile(compiler, visited=set(), obj_mfiles=[])

    # === PRIVATE ===
    
    def __init__(
        self,
        fcentral, dcentral, config,
        h_mfile, c_mfile, o_mfile, d_mfile, dfile
    ):
        self.fcentral = fcentral
        self.dcentral = dcentral
        self.config = config
        self.h_mfile = h_mfile
        self.c_mfile = c_mfile
        self.o_mfile = o_mfile
        self.d_mfile = d_mfile
        self.dfile = dfile
    
    def _should_compile(self):
        if self.c_mfile.exists:
            return (
                self.dfile is None or \
                self.dfile.should_compile()
            )
        else:
            return False
    
    def _compile(self, compiler):
        if self._should_compile():
            compiler.compile(inp=self.c_mfile.fpath, out=self.o_mfile.fpath)
            self.o_mfile.await_update()
            self.d_mfile.await_update()
            assert self.d_mfile.exists
            self.dfile = self.dcentral[self.d_mfile]
            assert not self._should_compile()
    
    def _rcompile(self, compiler, visited, obj_mfiles):
        
        if self.h_mfile not in visited:
            
            # Block cyclic future visits again
            visited.add(self.h_mfile)
            
            self._compile(compiler)
            if self.o_mfile.exists:
                obj_mfiles.append(self.o_mfile)
            
            if self.dfile is not None:
                for h_mfile in self.dfile.h_deps:
                    
                    #if h_mfile not in visited:
                        
                    srcfile = SrcFile.create(
                        fcentral=self.fcentral,
                        dcentral=self.dcentral,
                        config=self.config,
                        fpath=h_mfile.fpath
                    )
                    srcfile._rcompile(compiler, visited, obj_mfiles)
            
        assert len(obj_mfiles) == len(set(obj_mfiles))
        return obj_mfiles
