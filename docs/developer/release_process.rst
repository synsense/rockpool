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
