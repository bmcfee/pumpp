# pumpp
[![GitHub license](https://img.shields.io/badge/license-ISC-blue.svg)](https://raw.githubusercontent.com/bmcfee/pumpp/master/LICENSE)
[![CI](https://github.com/bmcfee/pumpp/actions/workflows/ci.yml/badge.svg)](https://github.com/bmcfee/pumpp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bmcfee/pumpp/branch/main/graph/badge.svg?token=i7YxIwweBQ)](https://codecov.io/gh/bmcfee/pumpp)
[![Documentation Status](http://readthedocs.org/projects/pumpp/badge/?version=latest)](http://pumpp.readthedocs.io/en/latest/?badge=latest)



practically universal music pre-processor

### pumpp up the jams

The goal of this package is to make it easy to convert pairs of `(audio, jams)` into data that can
be easily consumed by statistical algorithms.  Some desired features:

- Converting tags to sparse encoding vectors
- Sampling `(start, end, label)` to frame-level annotations at a specific frame rate
- Extracting input features (eg, Mel spectra or CQT) from audio
- Converting between annotation spaces for a given task

## Example usage

```python

>>> import jams
>>> import pumpp

>>> audio_f = '/path/to/audio/myfile.ogg'
>>> jams_f = '/path/to/annotations/myfile.jamz'

>>> # Set up sampling and frame rate parameters
>>> sr, hop_length = 44100, 512

>>> # Create a feature extraction object
>>> p_cqt = pumpp.feature.CQT(name='cqt', sr=sr, hop_length=hop_length)

>>> # Create some annotation extractors
>>> p_beat = pumpp.task.BeatTransformer(sr=sr, hop_length=hop_length)
>>> p_chord = pumpp.task.SimpleChordTransformer(sr=sr, hop_length=hop_length)

>>> # Collect the operators in a pump
>>> pump = pumpp.Pump(p_cqt, p_beat, p_chord)

>>> # Apply the extractors to generate training data
>>> data = pump(audio_f=audio_f, jam=jams_fjams_f)

>>> # Or test data
>>> test_data = pump(audio_f='/my/test/audio.ogg')

>>> # Or in-memory
>>> y, sr = librosa.load(audio_f)
>>> test_data = pump(y=y, sr=sr)
```
