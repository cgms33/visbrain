.. -*- mode: rst -*-

Visbrain
########

.. figure::  https://github.com/EtienneCmb/visbrain/blob/master/docs/_static/ico/visbrain.png
   :align:   center

**Visbrain** is an open-source python 3 package dedicated to brain signals visualization. It is based on top of `VisPy <http://vispy.org/>`_ and PyQt and is distributed under the 3-Clause BSD license. We also provide an on line `documentation <http://visbrain.org>`_, `examples and datasets <http://visbrain.org/auto_examples/>`_ and can also be downloaded from `PyPi <https://pypi.python.org/pypi/visbrain/>`_.

**Important**: This fork is a *lightweight* release of the Sleep module of Visbrain. Therefore, this fork only contains the source code for the Sleep module. There are also some notable differences between this fork and the main release of Visbrain:

1. The downsampling / filtering default methods are based on MNE
2. The default format for hypnogram export is in point-per-second.
3. This version works directly with MNE raw objects.
4. The lspopt dependency (multitaper spectrogram) is directly installed using pip.

Important links
===============

* Official source code repository : https://github.com/EtienneCmb/visbrain
* Online documentation : http://visbrain.org
* Visbrain `chat room <https://gitter.im/visbrain-python/chatroom?utm_source=share-link&utm_medium=link&utm_campaign=share-link>`_


Installation
============

Dependencies
------------

Visbrain requires :

* NumPy >= 1.13
* SciPy
* VisPy >= 0.5.3
* Matplotlib >= 1.5.5
* MNE >= 0.16
* Pandas >= 0.22
* PyQt5
* Pillow
* PyOpenGL
* lspopt

User installation
-----------------

Install Visbrain :

.. code-block:: shell

    pip install -e git+https://github.com/raphaelvallat/visbrain.git
