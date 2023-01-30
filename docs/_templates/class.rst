{{ fullname }}
==================================================================================================================================================================================================================================================

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :private-members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes overview

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block methods %}

   {% if methods %}
   .. rubric:: Methods overview

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

