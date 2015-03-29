# RITSAR
Synthetic Aperture Radar (SAR) Image Processing Toolbox for Python

This is an initial version of a SAR image processing toolbox for Python. The SciPy core libraries are required. The package was developed using the Anaconda distribution which comes with the SciPy core libraries.  Anaconda can be downloaded for free from https://store.continuum.io/cshop/anaconda/ . If using the omega-k algorithm, OpenCV is also required. Instructions for installing OpenCV for Python can be found at  https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup .  To get started, update the Python Dictionaries located in the parameters folder or leave the default values in place. In main.py in the top level directory, comment out those algorithms you do not wish to use.

Current capabilities include modeling the phase history for a collection of point targets as well as processing phase histories using the polar format, backprojection, and omega-k algorithms.

Over the coming months, I will update this readme with more detailed instructions as well as interfaces for AFRL Gotcha data and DIRSIG SAR simulated data.

To install, first download and unzip the repository.  Then from the command line, go to the unzipped directory and type "python setup.py install".  To uninstall, simply remove the ritsar directory.  This can be done by "rm -rf (Python Directory)/Libs/site-packages/ritsar" for an anaconda distribution of python.

If anyone is interested in collaborating, I can be reached at dm6718@g.rit.edu. Ideas on how to incorporate a GUI would be greatly appreciated.
