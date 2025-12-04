from setuptools import setup, Extension
import pybind11
import sys
import os 



# FFTW3 path
FFTW3_INCLUDE_DIR = '/usr/local/include'  # FFTW3 include
FFTW3_LIBRARY_DIR = '/usr/local/lib'      # FFTW3 library
FFTW3_LIBRARY_NAME = 'fftw3f'             # Linux/macOS，float version

#C++ extensions

ext_modules = [
    Extension(
        name="HkVp_multilayer.hk_stacking",  # Python module name
        sources=[
            "src/cpp_ext/mathfuns.cpp" ,      #functions
            "src/cpp_ext/hk_stacking.cpp",    # class implement
            "src/cpp_ext/bindings.cpp"       # pybind11 
        ],
        include_dirs=[
            "src/cpp_ext/include",               # self-defined include path in this package
            pybind11.get_include(),   # pybind11 
            FFTW3_INCLUDE_DIR         # FFTW3 
        ],
        library_dirs=[
            FFTW3_LIBRARY_DIR         # FFTW3 
        ],
        libraries=[
            FFTW3_LIBRARY_NAME        # FFTW3 
        ],  
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-fPIC"
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]  # NumPy 
    )
]


setup(
    name="HkVp_multilayer",
    version="0.1.0",
    description="HkVp_stacking for multilayer (C++ with pybind11)",
    author="Wentao Li",
    author_email="liwentao181@mails.ucas.ac.cn",
    install_requires=["pybind11>=2.6.0", "numpy>=1.15.0"], 
    python_requires=">=3.6",
    packages=['HkVp_multilayer'],
    package_dir={
        'HkVp_multilayer':'src'
    },
    scripts=[
    ],
    package_data={
        'HkVp_multilayer':['defaults/*','*.so','*.pyi','py.typed']
    },
    include_package_data=True,
    ext_modules=ext_modules,
    zip_safe=False 
)