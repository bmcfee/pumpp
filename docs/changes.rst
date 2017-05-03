Changes
-------

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
