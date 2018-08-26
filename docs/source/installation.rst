Installation
============
Use a **Python 3.7** environment for development.

Creating environment with Conda (example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having a Conda distribution (i.e. Anacoda, Minicoda etc) create environment like::

    conda create --name pyalcs python=3.7

Then activate it with::

    source activate pyalcs

Dependencies
^^^^^^^^^^^^
You should be fine with::

    pip install -r requirements.txt
    pip install -r requirements-integrations.txt

In case of troubles see ``Dockerfile`` and ``.travis.yml`` how the project is built from scratch.

Launching example integrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I assuming you are inside the virtual environment created before. In order to run the integrations from the console you need to specify Python PATH to use the currently checked-out version of ``pyalcs`` library::

    PYTHONPATH=<PATH_TO_MODULE> python examples/acs2/maze/acs2_in_maze.py

Interactive notebooks
^^^^^^^^^^^^^^^^^^^^^
Start the Jupyter notebook locally with::

  make notebook

Open the browser at ``localhost:8888`` and examine files inside ``notebooks/`` directory.
