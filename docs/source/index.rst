FRED tools v. |fredtoolsVersion| documentation
================================================================

FRED tools is a collection of python functions for image manipulation and analysis. The basic methods have been developed for analysis of the images produced by the `Monte Carlo FRED <http://www.fred-mc.org>`_ in MetaImage format (\*.mha, \*.mhd), but they can be applied for images in other formats, e.g. dicom.

Basic Concept
----------------------------
The image in the FRED tools is understood to mean scalar or vector image of any dimension from 2D to 5D. All the images read or processed by the functions are `SimpleITK <https://simpleitk.readthedocs.io>`_ images, and any SimpleITK routines can be applied. Check `SimpleITK filters <https://simpleitk.readthedocs.io/en/master/filters.html>`_ for available methods of filtering, registration, etc. Because the smallest image dimension supported by SimpleITK is 2D, line profiles, i.e. 1D images are saved as SimpleITK images with the size of only one of the dimension different than one.

Most of the functions have a bool argument ``displayInfo`` (default ``displayInfo=False``) which can be used for displaying a summary of the function results.

Documentation and tutorial
----------------------------
The documentation of the functions implemented in FRED tools is available at `jasqs.github.io/FREDtools <https://jasqs.github.io/FREDtools/>`_, at `fredtools.ifj.edu.pl <https://fredtools.ifj.edu.pl>`_ which forwards there, or on the `GitHub repository <https://github.com/jasqs/FREDtools>`_.

A simple `tutorial <https://github.com/jasqs/FREDtools/blob/main/examples/FREDtools%20Tutorial.ipynb>`_ has been prepared to help with starting using FRED tools.

Installation
----------------------------
FRED tools is published on `PyPI <https://pypi.org/project/fredtools/>`_ and is installed with pip, the package installer of Python.

Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FRED tools needs a 64-bit Python 3.11, 3.12, 3.13, 3.14 or 3.15 on Linux, Windows or macOS. With Python 3.10 or older, pip does not report an error but installs the old FRED tools 0.7.28, the last version that supported those Python versions.

pip installs the packages needed by FRED tools (e.g. numpy, scipy, SimpleITK, itk and pydicom) together with it. They are downloaded from PyPI as precompiled packages, so the installation needs access to the internet (directly, through a proxy or from a PyPI mirror) but no compiler.

Some parts of FRED tools have additional requirements:

- **Gamma index.** The gamma index is calculated by libFredGI, a compiled library distributed with FRED tools for:

  - Linux on x86_64 processors, with glibc 2.28 or newer (e.g. RHEL/Rocky 8, Ubuntu 20.04, Debian 10 and newer),
  - Windows 10 or newer on x86_64 processors,
  - macOS 11 or newer, on Apple Silicon and Intel processors.

- **GPU.** The influence matrices of FRED can be loaded into the memory of an NVIDIA graphics card (``getInmFREDSparse`` with ``interpreter="cupy"``) and summed there (``inmSumVec``, ``inmSumImg``). This needs:

  - an NVIDIA graphics card with enough memory for the influence matrix, and a driver supporting CUDA 12 or 13,
  - the CUDA Toolkit of the same version (12 or 13) installed on the system, or instead the CUDA libraries from PyPI, installed with ``pip install "cupy-cuda12x[ctk]"`` (or ``"cupy-cuda13x[ctk]"``),
  - Linux or Windows (not macOS) and Python 3.11 to 3.14.

  Without a GPU, these functions run on the CPU. The installation is described in GPU support below.

- **FRED simulations.** ``runFRED`` and the functions checking the installed FRED versions run the Monte Carlo FRED, which must be installed with the ``fred`` command available in the terminal. Reading and analysing FRED results does not need FRED.
- **Monte Carlo extra.** The functions for GATE and opengate, the Landau-Vavilov fits and the proton plan optimisation need the packages of the ``mc`` extra (Python 3.11 to 3.14), described in Monte Carlo extra below.
- **Jupyter.** The tutorials are Jupyter notebooks. Jupyter is not installed with FRED tools, install it with ``pip install jupyterlab ipympl``. The ipympl package is needed for the interactive mode of ``showSlices`` (``%matplotlib widget`` in the notebook).

Install FRED tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   pip install fredtools

This installs FRED tools with all the packages needed by most of its functions. To check that it works and which version is installed:

.. code:: bash

   python -c "import fredtools; print(fredtools.__version__)"

To update FRED tools to the newest version:

.. code:: bash

   pip install --upgrade fredtools

To install an exact version, e.g. to repeat an analysis done with that version:

.. code:: bash

   pip install fredtools==0.9.0

Monte Carlo extra
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A few functions need more packages, which are not installed by default: the readers of the GATE and opengate simulation results, the opengate Monte Carlo simulation, the Landau-Vavilov fits and the proton plan optimisation. To install FRED tools together with those packages, use the ``mc`` extra:

.. code:: bash

   pip install "fredtools[mc]"

The quotes are needed, because some terminals (e.g. zsh, the default on macOS) treat square brackets specially. Without the extra, everything else works and only those functions stop with an error like ``No module named 'opengate'``.

The ``mc`` extra needs Python 3.11 to 3.14 for now, because opengate and pyamtrack are not yet available for Python 3.15. opengate also needs a recent system: Linux with glibc 2.34 or newer (e.g. Ubuntu 22.04 or newer), macOS 15 or newer, or Windows. On an older system, install the other packages without opengate:

.. code:: bash

   pip install fredtools uproot landaupy pyamtrack scikit-image

GPU support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some functions can use an NVIDIA graphics card through the cupy package. Check the CUDA version of your driver, shown in the top right corner of the output of:

.. code:: bash

   nvidia-smi

and install the matching extra, for CUDA 12:

.. code:: bash

   pip install "fredtools[cuda12]"

or for CUDA 13:

.. code:: bash

   pip install "fredtools[cuda13]"

GPU support is not available on macOS and needs Python 3.11 to 3.14 for now.

Licence
----------------------------
FRED tools is distributed under the GNU General Public License v3 (see `LICENSE <https://github.com/jasqs/FREDtools/blob/main/LICENSE>`_), except the compiled gamma index libraries in ``fredtools/GammaIndex`` (libFredGI), which have their own notice: `LICENSE-libFredGI.md <https://github.com/jasqs/FREDtools/blob/main/fredtools/GammaIndex/LICENSE-libFredGI.md>`_.


Development
----------------------------
Everyone is invited to support the development of the FRED tools and to add functionalities needed for a specific application.
All the new functions or classes should be documented according to numpydoc style. Check `numpydoc <https://numpydoc.readthedocs.io>`_ and `napoleon <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy>`_ websites for the style guide and examples.

The FRED tools documentation is written in reStructuredText format and build with sphinx in version |sphinxVersion| using sphinx_rtd_theme template in version |RTDthemeVersion|.

Development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The newest, not yet released version can be installed directly from GitHub (a large download, because the repository includes the test data):

.. code:: bash

   pip install git+https://github.com/jasqs/FREDtools

To install a fixed version, add a commit, tag or branch, e.g. ``git+https://github.com/jasqs/FREDtools@<commit>``.

Working on FRED tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clone the repository and install FRED tools from the clone in editable mode (changes in the code take effect without reinstalling), together with the packages needed by the unit tests:

.. code:: bash

   git clone https://github.com/jasqs/FREDtools.git
   cd FREDtools
   python -m pip install --upgrade pip
   pip install -e . --group unittests

The ``--group`` option needs pip 25.1 or newer. Run the unit tests from the repository folder:

.. code:: bash

   python -m unittest discover -s unittests -t .

To build the documentation:

.. code:: bash

   pip install -e . --group docs
   sphinx-build -b html docs/source docs/html

The dependencies are defined in ``pyproject.toml``: the packages needed by FRED tools in ``dependencies``, the extras in ``[project.optional-dependencies]`` and the packages needed only for the development in ``[dependency-groups]``.

uv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FRED tools can also be installed with `uv <https://docs.astral.sh/uv/>`_:

.. code:: bash

   uv pip install fredtools
   uv pip install git+https://github.com/jasqs/FREDtools

In a uv project, a fixed commit of the development version can be declared in ``pyproject.toml`` as ``fredtools = { git = "https://github.com/jasqs/FREDtools", rev = "<commit>" }`` under ``[tool.uv.sources]``. In a clone of the repository, ``uv sync --group unittests`` creates the environment for the unit tests.

Release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A new version is published by raising ``_version`` in ``fredtools/_version.py`` and pushing it to the ``main`` branch. The GitHub workflow ``publish.yml`` then runs the unit tests, builds and deploys the documentation, publishes the new version on PyPI and creates the tag ``v<version>`` with a GitHub release. Pushes that do not change the version only run the tests and update the documentation.

.. toctree::
   :maxdepth: 1

   Introduction/Collaboration
   Introduction/Citation

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   Documentation/ImgIO
   Documentation/ImgAnalyse
   Documentation/ImgManipulate
   Documentation/BraggPeak
   Documentation/GammaIndex
   Documentation/MonteCarlo
   Documentation/ProtonOptimisation
   Documentation/Miscellaneous

Support
----------------------------
If You like the project and want to support our work you can buy as a coffee by clicking the image.

.. image:: buyCaffe.png
   :scale: 80%
   :alt: By a Caffee
   :align: center
   :target: https://buycoffee.to/fredtools
