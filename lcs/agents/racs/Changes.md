# Changes
## Representation
- _Don't care_ and _pass-through_ symbols (in ACS2 '#') are represented as fully ranged UBR.
- `complement_marks()` works if there is no previous value in the set

## Mark
- Mark holds a set of encoded perception values that holds bad states for classifier

## Effect
- `is_specializable` looks inside range.

# Thoughts
- Maybe effect could return just encoded value, instead of UBR...
