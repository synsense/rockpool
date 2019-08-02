API reference for class {{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :no-undoc-members:
   :inherited-members:
   :show-inheritance:

   .. automethod:: __init__

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}