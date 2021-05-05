Module {{ fullname }}
======={{ underline }}

.. contents::
    :local:

.. automodule:: {{ fullname }}
.. currentmodule:: {{ fullname }}

{% block functions %}
{% if functions %}

.. rubric:: Functions overview

.. autosummary::

{% for item in functions %}

  {{ item }}

{% endfor %}

.. rubric:: Functions

{%- for function in functions -%}
{%- if not function.startswith('_') %}
.. autofunction:: {{ function }}
{%- endif -%}
{%- endfor -%}

{% endif %}
{% endblock %}

{% if classes -%}
.. rubric:: Classes

{%- for class in classes -%}
{%- if not class.startswith('_') %}
.. autoclass:: {{ class }}
   :members:
{%- endif -%}
{%- endfor -%}
{%- endif %}
