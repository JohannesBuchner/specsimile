=======
emulate
=======

Neural network emulator for spectral models in physics.

About
-----

This package approximates the spectrum of a parametric physics simulator.
The approximation can be a neural network, or, through a intermediate parametric approximate 
function.

Example: The true physics model is a black body with an area A and temperature T (two parameters).
Between 400nm and 500nm, we approximate this model with a power law (normalisation N at 450nm, index p)
with two parameters. emulate learns the best mapping (A,T)->(N,p) to accurately reproduce
the spectrum.

The trained emulator is reusable and has infinite resolution.


You can help by testing emulate and reporting issues. Code contributions are welcome.
See the `Contributing page <https://johannesbuchner.github.io/emulate/contributing.html>`_.

.. image:: https://img.shields.io/pypi/v/emulate.svg
        :target: https://pypi.python.org/pypi/emulate

.. image:: https://github.com/JohannesBuchner/emulate/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/JohannesBuchner/emulate/actions/workflows/tests.yml

.. image:: https://coveralls.io/repos/github/JohannesBuchner/emulate/badge.svg?branch=master
        :target: https://coveralls.io/github/JohannesBuchner/emulate?branch=master

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/emulate/
        :alt: Documentation Status

Usage
^^^^^

Read the full documentation at:

https://johannesbuchner.github.io/emulate/


Licence
^^^^^^^

GPLv3 (see LICENCE file). If you require another license, please contact me.

Icon made by `Vecteezy <https://www.flaticon.com/authors/smashicons>`_ from `Flaticon <https://www.flaticon.com/>`_ .

