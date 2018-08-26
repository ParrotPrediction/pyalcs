Maze statistics
===============
After running an example integration, say ``acs2_in_maze.py``, here's what the
output tells you:

Agent stats
^^^^^^^^^^^
See ``lcs.agents.acs2.ACS2``

 * ``population``: number of classifiers in the population
 * ``numerosity``: sum of numerosities of all classifiers in the population
 * ``reliable``: number of reliable classifiers in the population
 * ``fitness``: average classifier fitness in the population
 * ``trial``: trial number
 * ``steps``: number of steps in this trial
 * ``total_steps``: number of steps in all trials so far

Environment stats
^^^^^^^^^^^^^^^^^
There are currently no environment statistics for maze environment.

Performance stats
^^^^^^^^^^^^^^^^^

 * ``knowledge``: As defined in
   ``examples.acs2.maze.utils.calculate_performance()``:
   If any of the reliable classifiers successfully predicts a transition, we
   say that the transition is anticipated correctly.  This is a percentage of
   correctly anticipated transitions among all possible transitions.
