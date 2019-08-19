from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

setup(
    cmdclass = {'build_ext':build_ext},

    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['PrepareBatchGraph.pyx','./src/lib/PrepareBatchGraph.cpp','./src/lib/utils.cpp','./src/lib/graph.cpp','./src/lib/graph_struct.cpp','./src/lib/graphUtil.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph', sources=['graph.pyx', './src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('metrics', sources=['metrics.pyx', './src/lib/metrics.cpp', './src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11'], include_dirs=[np.get_include()]),
                    Extension('utils', sources=['utils.pyx', './src/lib/utils.cpp', './src/lib/graph.cpp','./src/lib/graphUtil.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['graph_struct.pyx', './src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11'])
                   ])
