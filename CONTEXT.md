# Data Cleaning Agent

This context describes the language around transforming source data into cleaned
data while preserving enough identity to explain what changed.

## Language

**Source Row Identity**:
A stable identity assigned to each source row before cleaning, so cleaned data can
be aligned back to the original input after values, order, or row count change.
Every cleaning flow carries Source Row Identity for its source rows. Source Row
Identity is internal and is not a Protected Column. Each Source Row Identity maps
to at most one cleaned row.
_Avoid_: Raw row identity, row id, synthetic row id

**Protected Column**:
A source-data column that cleaning must preserve as user data unless the user
explicitly asks to change it. A Protected Column may follow standard label
normalization, but it is not dropped, stripped destructively, coerced, imputed,
or used as a dedupe comparison target.
_Avoid_: Keep-list column, excluded column

**Cleaning Run**:
One execution over source data that carries Source Row Identity from source
preparation through cleaned data, preview, outcome facts, and export.
_Avoid_: Session, context

## Example Dialogue

Developer: "Why do we need Source Row Identity if the cleaned data has fewer rows?"

Domain expert: "Because the removed rows still need to be explained against the
source data they came from."

Developer: "Can cleaning coerce a Protected Column if it looks numeric?"

Domain expert: "No. Protected Column means preserve it as source data unless the
user explicitly asks to change it."

Developer: "Where does Source Row Identity live while cleaning is in progress?"

Domain expert: "In the Cleaning Run. It follows the source data through cleaning
and is removed only when export data is produced."
