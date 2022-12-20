Notes for developers
====================

Checklist for releasing a new |project| version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Make a release candidate branch ``rc/<version>`` from ``develop`` branch on internal gitlab
- Make sure the version number in ``version.py`` has been bumped. We use semantic versioning: ``major.minor.maintenance.(postN)``, where ``(postN)`` is only for hotfixes
- Merge ``master`` into ``rc/...`` to make sure all changes are merged
- Push ``rc/..`` to ``origin``
- Make a merge request from ``rc/...`` into ``master``
- Get all primary developers to review the merge request, ensure that all suggested modifications are included
- Ensure that all pipelines pass, **including manual pipelines**
- Update ``CHANGELOG.md`` using ``git log X..Y --oneline``
- Once the merge has succeeded, delete the ``rc/...`` branch
- Make and push a tag to the ``master`` branch for the new version (i.e. "vX.Y.Z")
- Once all CI tasks have succeeded, a manual CI task "pypi_deploy" will be available. Run this task to deploy to PyPI. **This task must be run from the internal Rockpool repository**
- A pull request for the `conda feedstock <https://github.com/ai-cortex/rockpool-feedstock>`_ should be created automatically by a conda-forge bot. Check and merge this PR to bump the version on ``conda-forge``
- Merge ``master`` back into ``develop``
- Bump the version number in the ``develop`` branch to something like "vX.Y.Z.dev"

Headings for ``CHANGELOG.md``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ### Added
    ### Changed
    ### Fixed
    ### Deprecated
    ### Removed
    ### Security


Questions to resolve for each merge request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During code review for each merge request, make sure all of the following questions are addresssed.
If any boxes are not ticked, there should be a good reason why not.

- [ ] Are there user-facing tutorials?
    - what does the feature do? how to use it?
    - what are the limitations of the feature? what it doesn't do?
    - troubleshooting guide: what to do in case of issues/bugs/problems?
    - FAQ: what are the most common mistakes or confusions?
- [ ] Is each tutorial minimal and relatively self-contained?
- [ ] Is each tutorial didactic?
- [ ] Is the developer documentation updated with any new modules or changes?
- [ ] Are the packages / tutorials integrated into the documentation TOC?
- [ ] Are all class / method / function / file links in the tutorials working correctly?
- [ ] Do all user-facing class attributes have docstrings in `__init__()`?
- [ ] Do all non-user-facing class attributes have a leading underscore? (e.g. `._attr`)
- [ ] Are all user-facing class attributes either `Parameter`, `State` or `SimulationParameter` objects in `__init__()`?
- [ ] Do Rockpool modules obey the standard Rockpool API for `.evolve()`?
- [ ] `.as_graph()` implemented, if at all possible, for any new modules
- [ ] Does every `__init__.py` have an initial docstring block, describing what the package / sub-package does and contains?
- [ ] Are there tests implemented?
    - [ ] unit tests?
    - [ ] integration tests?
    - [ ] interface tests?
    - [ ] performance tests?
    - [ ] code coverage measured?
- [ ] Is `CHANGELOG.md` updated?
- [ ] Is the implementation minimal? i.e. the simplest possible way of implementing the functionality
