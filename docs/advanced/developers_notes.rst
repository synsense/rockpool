Notes for developers
====================

Checklist for releasing a new |project| version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Make sure the version number in ``version.py`` has been bumped. We use semantic versioning: ``major.minor.maintenance.(postN)``, where ``(postN)`` is only for hotfixes
- Make a release candidate branch ``rc/<version>`` from ``develop`` branch on internal gitlab
- Merge ``master`` into ``rc/...`` to make sure all changes are merged
- Make a merge request from ``rc/...`` into ``master``
- Get all primary developers to review the merge request, ensure that all suggested modifications are included
- Ensure that all pipelines pass, **including manual pipelines**
- Update ``CHANGELOG.md`` using ``git log X..Y --oneline``

Headings for ``CHANGELOG.md``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ### Added
    ### Changed
    ### Fixed
    ### Deprecated
    ### Removed
    ### Security
