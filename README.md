# pumpp
[![GitHub license](https://img.shields.io/badge/license-ISC-blue.svg)](https://raw.githubusercontent.com/bmcfee/pumpp/master/LICENSE)
[![Build Status](https://travis-ci.org/bmcfee/pumpp.svg?branch=master)](https://travis-ci.org/bmcfee/pumpp)
[![Coverage Status](https://coveralls.io/repos/github/bmcfee/pumpp/badge.svg?branch=master)](https://coveralls.io/github/bmcfee/pumpp?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pumpp/badge/?version=latest)](http://pumpp.readthedocs.org/en/latest/?badge=latest)



practically universal music pre-processor

### pumpp up the jams

The goal of this package is to make it easy to convert pairs of `(audio, jams)` into data that can
be easily consumed by statistical algorithms.  Some desired features:

- Converting tags to sparse encoding vectors
- Sampling `(start, end, label)` to frame-level annotations at a specific sampling rate
- Extracting first-level features (eg, Mel spectra or CQT) from audio
- Aligning and storing the results in a simple data structure (npz, hdf5)
- Converting between annotation spaces for a given task
- Helper variables for semi-supervised learning
