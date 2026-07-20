.. _contributing:

Contributing
############

.. _gh-discussions: https://github.com/cako/curvelets/discussions
.. _gh-discussions-feature-requests: https://github.com/cako/curvelets/discussions/new?category=feature-requests-and-ideas
.. _gh-discussions-bug-reports: https://github.com/cako/curvelets/discussions/new?category=report-issues
.. _gh-issues: https://github.com/cako/curvelets/issues
.. _numfocus-coc: https://numfocus.org/code-of-conduct
.. _conventional-commits: https://www.conventionalcommits.org/en/v1.0.0/#summary

We welcome and appreciate all contributions to the ``curvelets`` library! Please review the following guidelines before beginning your work.

To communicate with the core developers and maintainers, we request that you open a topic in our `GitHub Discussions <gh-discussions_>`_. Note that `as explained in Issue #55 <https://github.com/cako/curvelets/issues/55>`_, we ask external users and non-contributors to report bugs and request new features strictly via Discussions rather than opening direct issues on GitHub.

Furthermore, before participating, please read and abide by the `NUMFOCUS Code of Conduct <numfocus-coc_>`_. We expect all community members to uphold these standards.


Welcomed contributions
**********************

Bug reports
===========

If you encounter unexpected behavior while computing transforms with ``curvelets``, please report the bug via our `GitHub Discussion for bug reports <gh-discussions-bug-reports_>`_.

To help us diagnose and reproduce the problem quickly, please include:

* The exact name and version of your operating system.
* Detailed details regarding your Python environment and dependencies.
* A self-contained, minimal script or step-by-step instructions showing how to reproduce the bug from a clean installation.


New methods and features
========================

If you have ideas for new directional transforms, operators, or functional improvements, we invite you to open a `GitHub Discussion for new features <gh-discussions-feature-requests_>`_.

When proposing a new method or feature:

* Explain in detail the mathematical rationale and numerical behavior of the proposed method.
* Keep the initial implementation scope narrow to ensure that we can review and maintain the code effectively.
* Provide a minimal example that shows the intended usage and expected numerical output.


Fix issues
==========

We maintain a tracked backlog of validated bugs and planned features in `GitHub Issues <gh-issues_>`_. Note that these issues represent tasks that maintainers have triaged from community discussions.

We welcome contributors to inspect open issues and submit pull requests targeting our ``main`` branch to resolve them.


Add examples or improve documentation
=====================================

Writing new code is only one of many ways to strengthen the library. We highly encourage contributors to develop new gallery examples showing practical applications of existing methods, or to improve the clarity and completeness of our API documentation.


Step-by-step instructions for contributing
******************************************

When you are ready to contribute code or documentation, follow this workflow:

1. **Environment setup with uv**

   We use ``uv`` as our primary package and environment manager. First, clone the repository and install the package in development mode with all optional dependencies:

   .. code-block:: bash

      uv pip install -e .[torch]

   Next, install our development and testing dependencies:

   .. code-block:: bash

      uv pip install --group dev

2. **Branch creation**

   Create a dedicated feature branch starting from our latest ``main`` branch:

   .. code-block:: bash

      git checkout -b name-of-your-branch main

   You can now implement your modifications locally.

3. **Running tests**

   We use ``pytest`` managed via ``nox`` to test across multiple Python environments. You can run the test suite directly using our convenient ``Makefile`` shortcut:

   .. code-block:: bash

      make test

   Note that this command delegates to ``uv run nox -s tests``. You can also invoke specific sessions explicitly if needed.

4. **Running static analysis and linting**

   We maintain strict standards for code formatting, linting, and static type safety (``ruff``, ``ty``, ``mypy``). To verify that your code complies with our standards, run:

   .. code-block:: bash

      make lint

   This shortcut executes ``uv run nox -s lint`` across the codebase. Therefore, we strongly recommend running ``make lint`` before submitting your code for review.

5. **Updating documentation**

   If your contribution modifies any API signature or introduces new functionality, please update the documentation and gallery scripts accordingly. To build the HTML documentation locally:

   .. code-block:: bash

      make doc

   This command runs ``uv run nox -s docs`` and generates our Sphinx documentation gallery in ``docs/_build/html``.

6. **Committing your changes**

   Once you have verified that all tests and linting checks pass, stage and commit your changes:

   .. code-block:: bash

      git add .
      git commit -m "feat: detailed description of your changes"
      git push -u origin name-of-your-branch

   We recommend following `Conventional Commits <conventional-commits_>`_ formatting for all commit messages.

7. **Submitting a pull request**

   Finally, open a pull request on GitHub comparing your feature branch against our ``main`` branch.


Pull request guidelines
***********************

Before submitting a pull request, verify that your contribution meets the following criteria:

1. **Comprehensive test coverage**: Include unit tests verifying both forward and backward numerical precision for all newly added routines.
2. **Documentation synchronization**: Ensure that docstrings and gallery tutorials accurately reflect any changes to function arguments or behavior.
3. **Clean verification**: Confirm that your branch passes all automated checks by running ``make test`` and ``make lint`` locally without errors or warnings.


Project structure
*****************

We organize the ``curvelets`` repository into the following functional directories:

* ``src/curvelets``: Core Python package containing our Uniform Discrete Curvelet Transform (UDCT) implementations for NumPy and PyTorch.
* ``tests``: Unit and integration test suites executed via ``pytest``.
* ``testdata``: Standard reference images and synthetic datasets used across our test assertions and gallery examples.
* ``docs``: Sphinx configuration, ReStructuredText manuals, and bibliography files (``references.bib``).
* ``examples``: Python scripts formatted for ``sphinx-gallery`` that generate our visual tutorials and verification charts.


Development tools
*****************

We rely on modern, high-performance tooling to maintain our developer workflow:

* **uv**: Fast Python package and virtual environment manager.
* **nox**: Automated task runner managing isolated testing and documentation sessions.
* **pre-commit**: Git hooks enforcing rapid formatting and checks prior to commits.
* **ruff**: High-speed Python linter and code formatter.
* **ty**: Ultra-fast Rust-based static type checker.
* **mypy**: Standard static type checker ensuring structural typing integrity across modules.
* **pytest**: Comprehensive testing framework used for numerical precision and gradient verification.
* **sphinx**: Documentation engine powering our HTML manual and gallery build.


Automated workflows (GitHub Actions)
************************************

We automate quality assurance, multi-platform testing, and package distribution using GitHub Actions configured in ``.github/workflows/``:

* **Continuous Integration (``ci.yml``)**: Whenever you push commits or open a pull request, our CI pipeline automatically verifies your changes across a comprehensive build matrix:

  * **Code formatting and linting**: A dedicated job checks all files against our pre-commit hooks and runs ``pylint`` via ``nox`` to guarantee formatting consistency.
  * **Multi-platform and multi-version testing**: We run our complete ``pytest`` suite across Linux (Ubuntu), macOS, and Windows for Python versions ``3.9`` through ``3.14`` (including free-threaded ``3.14t`` and ``PyPy 3.10``).
  * **Coverage tracking**: Test execution generates XML coverage reports that are automatically uploaded and analyzed via Codecov.

* **Continuous Deployment (``cd.yml``)**: When maintainers publish a new release on GitHub, our CD workflow verifies package integrity using ``build-and-inspect-python-package`` and automatically publishes the built artifacts to PyPI using secure OIDC authentication (``pypa/gh-action-pypi-publish``).

Therefore, when you submit a pull request, you can monitor these automated workflow checks directly on the GitHub PR page to verify that your branch works seamlessly across all supported environments.
