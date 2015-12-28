# RITSAR
Synthetic Aperture Radar (SAR) Image Processing Toolbox for Python

Before installation, please make sure you have the following:
- SciPy. Comes with many Python distributions such as Enthought Canopy, Python(x,y), and Anaconda.  Development was done using the Anaconda distribution which can be downloaded for free from https://store.continuum.io/cshop/anaconda/. 
- OpenCV (optional). If using the omega-k algorithm, OpenCV is required. Instructions for installing OpenCV for Python can be found at  https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup.
- Spectral (optional).  Needed to interface with .envi files.  Can be installed from the command line using

  $ pip install spectral

  alternatively, Spectral can be downloaded here: http://www.spectralpython.net/ 
  
To get started, first make sure your SciPy and NumPy libraries are up-to-date.  With Anaconda, this can be done by typing the following into a terminal or command prompt:

$ conda update conda

$ conda update anaconda

Once you've ensured the required libraries are up-to-date, download the zip file and extract it to a directory that from here on will be referred to as \<ritsar_dir\>.  Open up a command line or terminal and type:

$ cd \<ritsar_dir\>

$ python setup.py install

then...

$ cd ./examples

$ ipython --pylab

From the ipython console, type:

In [1]: %run FFBPmp_demo

In [2]: import matplotlib.pylab as plt; plt.show()

or run any other demo.  Alternatively, you can open up the demos in an IDE of your choice to experiment with the different options available.

Current capabilities include modeling the phase history for a collection of point targets as well as processing phase histories using the polar format, omega-k, backprojection, digitally spotlighted backprojection, fast-factorized backprojection, and fast-factorized backprojection with multi-processing algorithms.  Autofocusing can also be performed using the Phase Gradient Algorithm.  The current version can interface with AFRL Gotcha and DIRSIG data as well as a data set provided by Sandia.

Data included with this toolset includes a small subset of the AFRL Gotcha data provided by AFRL/SNA.  The full data set can be downloaded separately from https://www.sdms.afrl.af.mil/index.php?collection=gotcha after user registration.  Also included is a single dataset from Sandia National Labs.

If anyone is interested in collaborating, I can be reached at dm6718@g.rit.edu. Ideas on how to incorporate a GUI would be greatly appreciated.
