from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='Generalized-Rational-Variable-Projection-with-Application-in-ECG-Compression',
        version='0.1',
        packages=find_packages(where='code'),
        package_dir={'': 'code'},
        install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'PyWavelets',
            # Add other dependencies as needed
    ],
    extras_require={
            'testing': [
                'pytest>=6.0',
                'pytest-cov>=2.0',
                'tox>=3.24',
            ],
        },
    )