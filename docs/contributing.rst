.. _contributing:

Contributing
############

We love contributions! Please see the following sections for more information on how to contribute.

The best way to get in touch with the core developers and maintainers is to
open a new
`GitHub Discussion <https://github.com/cako/curvelets/discussions>`_.

Please do not open new *Issues* directly, open a `GitHub Discussion for feature requests <https://github.com/cako/curvelets/discussions/new?category=feature-requests-and-ideas>`_ instead.

Before you read on, please read the `NUMFOCUS Code of Conduct <https://numfocus.org/code-of-conduct>`_.
We expect all contributors to adhere to this code of conduct.

Welcomed contributions
**********************

Bug reports
===========

Found a bug? Report it `GitHub Discussion for bug reports <https://github.com/cako/curvelets/discussions/new?category=report-issues>`_.

If you find a bug, please report it including:

* Your operating system name and version.
* Detailed steps to reproduce the bug from a fresh environment.
* Please provide a minimal example to reproduce the bug.

New features
============

Have a new idea? Would like to propose a new feature? Open a new `GitHub Discussion for new features <https://github.com/cako/curvelets/discussions/new?category=feature-requests-and-ideas>`_.

If you are proposing a new feature, please explain in detail how it should work.
Keep the scope as narrow as possible, to make it easier to implement.

The features most likely to be implemented are those which contain a clear and concise description of the feature, a minimal example to reproduce the feature, and a clear and concise description of the expected behavior.

Fix issues
==========

A great way to contribute is to fix issues!
In `GitHub Issues <https://github.com/cako/curvelets/issues>`_ we track issues which have been triaged and reproduced by the maintainers.
Fixes should be submitted as pull requests to the main branch.


Step-by-step instructions for contributing
******************************************

Ready to contribute?

1. **Development installation**

   This project uses ``uv`` as the package manager. Install the package in
   development mode:

   .. code-block:: bash

      uv pip install -e .[torch]

   Install development dependencies:

   .. code-block:: bash

      uv pip install --group dev

2. **Create a branch for local development**

   Create a branch for your changes, usually starting from the most up-to-date ``main`` branch:

   .. code-block:: bash

      git checkout -b name-of-your-branch main

   Now you can make your changes locally.

3. **Run tests**

   When you're done making changes, check that your code passes all tests:

   .. code-block:: bash

      make test

   This will run tests across all supported Python versions (3.9-3.14).

4. **Run linting**

   Run the linter to check the quality of your code:

   .. code-block:: bash

      make lint

   This runs pre-commit hooks which include ruff, mypy, and other code quality
   checks. **Your code should ideally pass ``make lint`` before submitting a PR.**

5. **Update the docs**

   If you've added new functionality, update the documentation:

   .. code-block:: bash

      make doc

   This will regenerate the API documentation and build the docs.

6. **Commit your changes and push your branch**

   .. code-block:: bash

      git add .
      git commit -m "Your detailed description of your changes."
      git push -u origin name-of-your-branch

   We recommend using `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_
   to format your commit messages, but this is not enforced.

7. **Submit a pull request**

   Submit a pull request through the GitHub website.


Pull Request Guidelines
***********************

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have
   been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
3. Ensure that the updated code passes all tests (`make test`).
4. Ensure that the code passes linting (`make lint`).

If you have any questions or are unsure about something, please feel free to open a PR and request feedback.


Tools
*****

This project uses several tools for development:

* **uv**: Package manager and environment management
* **nox**: Task runner for running tests, linting, and building docs
* **pre-commit**: Git hooks for code quality checks
* **ruff**: Fast Python linter and formatter
* **mypy**: Static type checker
* **pytest**: Testing framework

All tools are configured in the project and will be automatically used when you
run ``make lint`` or ``make test``.
