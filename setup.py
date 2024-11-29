"""
Author:  Mohammad Daffa Robani <robani.daffa@gmail.com>
        Paul Saves <paul.saves@onera.fr>
        Remi Lafage <remi.lafage@onera.fr>
        Pramudita Satria Palar < pramsp@itb.ac.id>

This package is distributed under New BSD license.
"""

from setuptools import setup


# Import __version__ without importing the module in setup
exec(open("./smt_explainability/version.py").read())

metadata = dict(
    name="smt-explainability",
    version=__version__,  # noqa
    description="",
    long_description="",
    author="Daffa Robani et al.",
    author_email="robani.daffa@gmail.com",
    maintainer="Paul Saves",
    maintainer_email="paul.saves@onera.fr",
    license="BSD-3",
    packages=["smt_explainability","smt_explainability.pdp","smt_explainability.problems","smt_explainability.shap" ],
    install_requires=["smt>=2.8.0", "smt-design-space-ext>=0.3.0"],
    extras_require={},
    python_requires=">=3.9",
    zip_safe=False,
)

setup(**metadata)
