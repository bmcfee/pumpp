Changes
-------

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
