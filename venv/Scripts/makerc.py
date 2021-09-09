#!C:\Users\lenovo\PycharmProjects\pythonProject\khadija\venv\Scripts\python.exe

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--config", default="cmaker.config")
    parser.add_argument("--init", type=int, default=0)
    args = parser.parse_args()
    
    from cmaker.maker import Maker
    from cmaker.compiler import Compiler
    
    maker = Maker(args.config)
    
    if args.init:
        maker.config.write(args.config)
    else:
        if args.input is None or args.output is None:
            raise SystemExit("ERROR: Provide inputs and outputs.")
        else:
            try:
                maker.make(args.input, args.output)
            except Compiler.Error:
                raise SystemExit("ERROR: Early termination.")
