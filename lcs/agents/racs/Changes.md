# Changes
## Representation
- _Don't care_ and _pass-through_ symbols (in ACS2 '#') are represented as fully ranged UBR.
- `complement_marks()` works if there is no previous value in the set

## Classifier
- `Specialize` creates a fixed, narrow UBR like `UBR(4, 4)`. Later on during another processes it can be generalized more.

## Mark
- Mark holds a set of encoded perception values that holds bad states for classifier

## Effect
- `is_specializable` looks inside range.

# Thoughts
- Maybe effect could return just encoded value, instead of UBR...
- Specificity/generality should measure how wide is the UBR