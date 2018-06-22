Changes
-------

0.3.3
=====
- `#95`_ Data durations are now checked before sampling

.. _#95: https://github.com/bmcfee/pumpp/pull/95

0.3.2
=====
- `#91`_ JAMS annotations are now populated with confidence fields
- `#92`_ Pump objects can pretty-print in jupyter notebooks

.. _#91: https://github.com/bmcfee/pumpp/pull/91
.. _#92: https://github.com/bmcfee/pumpp/pull/92

0.3.1
=====
- `#88`_ Added time-position coding
- `#87`_ Added variable-length sampler

.. _#88: https://github.com/bmcfee/pumpp/pull/88
.. _#87: https://github.com/bmcfee/pumpp/pull/87

0.3.0
=====
- `#85`_ Fixed a bug in BeatPosition transforms
- `#84`_ Fixed a bug in the documentation build on readthedocs
- `#83`_ Fixed an off-by-one error in sampler
- `#81`_ Support multiple time-like dimensions in sampling
- `#80`_ Added `crop=` parameter to `Pump.transform`, which can slice temporal data down to a common duration.

.. _#85: https://github.com/bmcfee/pumpp/pull/85
.. _#84: https://github.com/bmcfee/pumpp/pull/84
.. _#83: https://github.com/bmcfee/pumpp/pull/83
.. _#81: https://github.com/bmcfee/pumpp/pull/81
.. _#80: https://github.com/bmcfee/pumpp/pull/80

0.2.4
=====
- `#76`_ Implemented the beat-position task

.. _#76: https://github.com/bmcfee/pumpp/pull/76


0.2.3
=====
- `#74`_ Implemented segmentation agreement task

.. _#74: https://github.com/bmcfee/pumpp/pull/74


0.2.2
=====

- `#70`_ Future-proofing against jams 0.3

.. _#70: https://github.com/bmcfee/pumpp/pull/70

0.2.1
=====

- `#68`_ Fixed a frame alignment error in task transformers
- `#66`_ Remove warnings for improperly cast STFT data

.. _#68: https://github.com/bmcfee/pumpp/pull/68
.. _#66: https://github.com/bmcfee/pumpp/pull/66

0.2.0
=====
- `#65`_ Removed old-style (function) transform API in favor of object interface
- `#65`_ Support in-memory analysis

.. _#65: https://github.com/bmcfee/pumpp/pull/65

0.1.5
=====
- `#61`_ Fixed an alignment bug in feature extractors

.. _#61: https://github.com/bmcfee/pumpp/pull/61

0.1.4
=====
- `#59`_ harmonic CQT
- `#58`_ Sparse chord output for chord labels
- `#57`_ Updated sampler bindings for Pump object

.. _#59: https://github.com/bmcfee/pumpp/pull/59
.. _#58: https://github.com/bmcfee/pumpp/pull/58
.. _#57: https://github.com/bmcfee/pumpp/pull/57

0.1.3
=====

- `#55`_ Refactored the `Sampler` class, added support for random states and the `SequentialSampler` class

.. _#55: https://github.com/bmcfee/pumpp/pull/55

0.1.2
=====

- `#51`_ Added named operator index to `Pump` objects

.. _#51: https://github.com/bmcfee/pumpp/pull/51

0.1.1
=====

- `#49`_ Added `Pump.layers` constructor for Keras layers on pump containers
- `#47`_ Fixed a bug in `Sampler` that caused a shape mismatch on input/output tensors
  when the input JAMS had multiple matching annotations for a given task.

.. _#49: https://github.com/bmcfee/pumpp/pull/49
.. _#47: https://github.com/bmcfee/pumpp/pull/47

0.1.0
=====

- Initial public release
