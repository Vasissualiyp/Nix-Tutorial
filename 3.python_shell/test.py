import sys
print(f"Python version: {sys.version}")

def check_version(package:str):
    try:
        exec(f"import {package}")
        exec(f"print('{package} version: ' + {package}.__version__)")
    except:
        print(f"No {package} found")
# numpy
check_version("numpy")
check_version("matplotlib")
check_version("pandas")
check_version("scipy")
check_version("astropy")

