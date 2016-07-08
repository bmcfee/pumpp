.. pumpp documentation master file, created by
   sphinx-quickstart on Thu Jul  7 21:27:51 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Practically Universal Music Pre-Processor
=========================================

Pumpp is designed to make it easy to convert pairs of `(audio, jams)` into data that can
be easily consumed by statistical algorithms.  Some desired features:

- Converting tags to sparse encoding vectors
- Sampling `(start, end, label)` to frame-level annotations at a specific sampling rate
- Extracting first-level features (eg, Mel spectra or CQT) from audio
- Aligning and storing the results in a simple data structure (npz, hdf5)
- Converting between annotation spaces for a given task
- Helper variables for semi-supervised learning


API
===
.. toctree::
    :maxdepth: 2

    api


Changes
=======
.. toctree::
    :maxdepth: 2

    changes

Contribute
==========
- `Issue Tracker <http://github.com/bmcfee/pumpp/issues>`_
- `Source Code <http://github.com/bmcfee/pumpp>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

