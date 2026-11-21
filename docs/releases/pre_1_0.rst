Before 1.0
==========

Alpha and beta releases, from the first PyPI upload to the last beta before 1.0.

0.2.0b1 (2025-12-28)
--------------------

**Monogenic transform.** (:pr:`38`)

- ``UDCT.forward_monogenic``, ``UDCT.backward_monogenic`` and ``UDCT.monogenic``, built on new Riesz filters, with the ``MUDCTCoefficients`` type.
- ``window_overlap`` is computed from the Nguyen–Chauris constraint when it is not given.
- ``high_frequency_mode`` accepts ``"curvelet"`` or ``"wavelet"``.
- New gallery examples: monogenic demos and a window-overlap heatmap.

0.1.0b1 to 0.1.0b4 (2025-12-27)
-------------------------------

First beta releases. (:pr:`34`)

- ``UDCT`` rewritten as modular code (transform, forward, backward and typing modules) with sparse window utilities.
- N-dimensional UDCT with exact invertibility, real and complex transform modes, and ``high_frequency_mode`` options.
- ``MeyerWavelet`` added; ``UDCT``, ``MeyerWavelet``, ``UDCTCoefficients`` and ``UDCTWindows`` are the public names.
- Type annotations throughout; Python 3.9 to 3.14 (including free-threaded 3.14t) supported.
- Documentation with a features table, FAQs, references and examples comparing curvelets and wavelets.

0.0.1-alpha to 0.0.6-alpha (2024-11-21 to 2024-11-25)
-----------------------------------------------------

- 0.0.1-alpha is the first test release and 0.0.2-alpha is the first upload to PyPI.
- Direction-dependent resolution: the number of wedges can differ per direction. (:pr:`15`)
- The ``transpose`` option was added and then removed again. (:pr:`16`)
- First documentation site, API reference and the single-disk example. (:pr:`18`)
