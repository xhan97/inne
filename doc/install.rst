#####################################
Installation & Setup
#####################################


Welcome
=======

Your first step is to install inne, which comes bundled as a pip package.


Virtual Environment
==================

We highly encourage you to install inne in a virtual environment. We like to use `Anaconda <https://docs.anaconda.com/anaconda/user-guide/getting-started/>`__ to manage our Python virtual environments.


Install with pip
================
When you're set with your environment, run:

.. code-block:: bash

    pip install inne

If you're feeling brave, feel free to install the bleeding edge: NOTE: Do so at your own risk; no guarantees given!

.. code-block:: bash

    pip install git+https://github.com/xhan97/inne.git@master  --upgrade

Alternatively download the package, install requirements, and manually run the installer:

.. code-block:: bash

    wget https://codeload.github.com/xhan97/inne/zip/refs/heads/master
    unzip inne-master.zip
    rm inne-master.zip
    cd inne-master

    pip install -r requirements.txt

    python setup.py install

Once the installation is completed, you can check whether the installation was successful through:

.. code-block:: python

    import inne
    print(inne.__version__)

