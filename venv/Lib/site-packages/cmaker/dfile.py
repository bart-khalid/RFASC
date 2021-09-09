class DFile:
    
    @staticmethod
    def parse(fcentral, mfile):
        fit = mfile.parse()
        
        head = next(fit)
        target, all_deps = head.split(": ")
        all_deps = _clean(all_deps)
        for subs in map(_clean, fit):
            all_deps.extend(subs)
        
        all_deps = list(filter(None, all_deps))
        
        target = fcentral[target]
        src = fcentral[all_deps[0]]
        h_deps = [fcentral[all_deps[i]]
            for i in range(1, len(all_deps))
        ]
        return DFile(mfile, target, src, h_deps)
    
    def __init__(self, d_mfile, target, src, h_deps):
        self.d_mfile = d_mfile
        self.target = target
        self.src = src
        self.h_deps = h_deps
    
    def recipe(self):
        tar = self.target.fpath
        src = self.src.fpath
        hed = " ".join([
            h.fpath for h in self.h_deps
        ])
        return "%s: %s %s" % (tar, src, hed)
    
    def should_compile(self):
        
        if not self.src.exists:
            return False
            
        else:
            
            if not self.target.exists:
                return True
            
            for h_mfile in self.h_deps:
                if self._check_header_age(h_mfile):
                    return True

            return self._check_src_age()
    
    # === PRIVATE ===
    
    def _check_header_age(self, h_mfile):
        return not h_mfile.exists or self._check_dep_age(h_mfile)

    def _check_dep_age(self, mfile):
        return self.target < mfile or self.d_mfile < mfile
    
    def _check_src_age(self):
        return self._check_dep_age(self.src)

# === PRIVATE ===

def _clean(line):
    return line.rstrip(" \\").rstrip().split(" ")
