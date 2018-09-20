Changes
=======
The following is and *in-progress* list of all modifications made to the ACS2:

Representation
^^^^^^^^^^^^^^
* ``Don't care`` and ``pass-through`` symbols (in ACS2 ``'#``') are represented as fully ranged UBR.
* ``complement_marks()`` works if there is no previous value in the set

Classifier
^^^^^^^^^^
* ``specialize()`` creates a fixed, narrow UBR like ``UBR(4, 4)``. Later on during another processes it can be generalized more.
* ``is_more_general`` looks at average ``cover_ratio``

Condition
^^^^^^^^^
* ``cover_ratio`` function,
* ``does_match_condition`` - analyzes incorporation (when doing subsumption)

Mark
^^^^
* Mark holds a set of encoded perception values that holds bad states for classifier

Effect
^^^^^^
* ``is_specializable`` looks inside range.


Components
^^^^^^^^^^
* ALP implemented
* Custom mutation in GA (condition and effect)

Thoughts
^^^^^^^^
* Maybe effect could return just encoded value, instead of UBR...
* Specificity/generality should measure how wide is the UBR
* Maybe ``u_max`` should hold information how specific condition should be (not just wildcards but spread)
* In the end of ALP phase we should perform something like classifier merge
* There is still very aggressive``generalize`` function in Condition part.
