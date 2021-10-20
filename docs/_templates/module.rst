Module {{ fullname }}
======={{ underline }}

.. contents::
    :local:

.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

{% block modules %}
{% if modules %}

.. rubric:: Submodules

.. autosummary::
  :template: module.rst
  :recursive:

{% for item in modules %}

  {{ item }}

{% endfor %}

{%- endif -%}

{% endblock %}


{% block functions %}
{% if functions %}

.. rubric:: Functions overview

.. autosummary::

{% for item in functions %}

  {{ item }}

{% endfor %}

{%- endif -%}

{% endblock %}


{% block classes %}
{% if classes %}

.. rubric:: Classes overview

.. autosummary::

{% for item in classes %}

  {{ item }}

{% endfor %}

{%- endif -%}
{% endblock %}



{% if functions -%}
.. rubric:: Functions

{%- for function in functions -%}
{%- if not function.startswith('_') %}
.. autofunction:: {{ function }}
{%- endif -%}
{%- endfor -%}

{% endif %}



{% if classes -%}
.. rubric:: Classes

{%- for class in classes -%}
{%- if not class.startswith('_') %}
.. autoclass:: {{ class }}
   :members:
{%- endif -%}
{%- endfor -%}
{%- endif %}
