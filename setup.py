from setuptools import setup, find_packages

setup(
    name="ctc_forced_alignment",
    version="0.1.0",
    description="Forced alignment using Omnilingual ASR CTC model",
    author="Your Name",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "align=align:main",
        ],
    },
)