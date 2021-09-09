import os

class Config:
    
    @staticmethod
    def parse(fpath):
        kwargs = {}
        if os.path.isfile(fpath):
            with open(fpath, "r") as f:
                for line in map(str.strip, f):
                    if line and line[0] != "#":
                        parts = line.split("=")
                        assert len(parts) > 1
                        k = parts[0]
                        v = "=".join(parts[1:])
                        kwargs[k.strip()] = v.strip()
        return Config(**kwargs)

    def write(self, fpath):
        data = "\n".join([
            " = ".join(item)
            for item in vars(self).items()
        ])
        with open(fpath, "w") as f:
            os.fsync(f.fileno())
            f.write(data)
            f.flush()
    
    def __init__(
        self,
        c_ext=".cpp",
        obj_dirsep=".",
        obj_dir="dump",
        compile_cmd="g++ -MMD -c {inp} -o {out}",
        combine_cmd="g++ -MMD {inp} -o {out}"
    ):
        self.c_ext = c_ext
        self.obj_dirsep = obj_dirsep
        self.obj_dir = obj_dir
        self.compile_cmd = compile_cmd
        self.combine_cmd = combine_cmd
