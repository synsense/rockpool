Before submitting this MR for review, please make sure:

- [ ] code builds clean without errors and (at least) no increase number of warnings
- [ ] documentation builds clean without errors and (at least) no increase number of warnings
- [ ] there are new documentation and tests about this feature as needed
- [ ] description of the modification was added to the CHANGELOG

## Description

What this MR does and why do we need it?
Please, describe code changes for reviewer.
List out the areas affected by your code changes.

* Is there any existing behavior change of other features due to this code change? If so, explain.

## Reviewing MR

During code review for each merge request, make sure all of the following questions are addresssed.
If any boxes are not ticked, there should be a good reason why not.

- [ ] Are there user-facing tutorials?
    - [ ] what does the feature do? how to use it?
    - [ ] what are the limitations of the feature? what it doesn't do?
    - [ ] troubleshooting guide: what to do in case of issues/bugs/problems?
    - [ ] FAQ: what are the most common mistakes or confusions?
    - [ ] Is each tutorial minimal and relatively self-contained?
    - [ ] Is each tutorial didactic?
    - [ ] Are all class / method / function / file links in the tutorials working correctly?
- [ ] Is the developer documentation updated with any new modules or changes?
- [ ] Are the packages / tutorials integrated into the documentation TOC?
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
    - [ ] Does it contains an one-line description of the functionality?
    - [ ] Does the dev version have been updated?
- [ ] Is ``version.py`` updated and is it matching ``CHANGELOG.md``?
- [ ] Is the implementation minimal? i.e. the simplest possible way of implementing the functionality
