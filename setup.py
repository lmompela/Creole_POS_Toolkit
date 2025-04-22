from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='pos_tagger',
    version='0.1.0',
    description='Transformer-based POS tagging toolkit for CoNLL-U formatted data',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Ludovic Vetea Mompelat',
    author_email='lvm861@miami.edu',
    url='https://github.com/lmompela/Creole_POS_Toolkit',
    license='MIT',

    packages=find_packages(exclude=['tests', 'scripts']),
    include_package_data=True,

    # Install scripts for CLI
    scripts=[
        'scripts/train.py',
        'scripts/predict.py',
        'scripts/analyze.py'
    ],

    install_requires=install_requires,

    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Text Processing :: Linguistic'
    ],
)
