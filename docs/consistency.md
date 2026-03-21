# Consistency with original rdchiral

The intention of this fork is to maintain consistency with the upstream `rdchiral` library while improving performance through static typing and compilation with `mypyc`. Simply install this library and use it as a drop-in replacement for `rdchiral`. Most changes in this repository are focused on improving performance relative to upstream `rdchiral`. However, a small number of behavioral changes also exist, described below.

The table below summarizes agreement with upstream `rdchiral` for a fixed, deterministic set of template applications.

| library | number identical outcomes |
| --- | --- |
| purepy | 9987 |
| mypyc | 9987 |
| cpp | 9954 |

## Behavioral differences

Relevant upstream changes and discussion:

- https://github.com/connorcoley/rdchiral/pull/40: Fixes incorrect cis/trans outcomes for conjugated systems that could previously depend on atom numbering. In particular, when a template only specifies part of a conjugated system, the copied double-bond stereo directions may need to be reversed consistently.
- https://github.com/connorcoley/rdchiral/pull/31: Template extraction corner cases could return `None` instead of a dict, leading to inconsistent downstream behavior (and possible `AttributeError`s for callers expecting a mapping). This change makes the return type consistent.
- https://github.com/connorcoley/rdchiral/commit/78bbafaba040678b957497e7f2638e935104e3d7: Extends template extraction to support a configurable fragment `radius` and an option to disable matching/including “special groups” (`no_special_groups`), which can change which atoms are included in extracted fragments.
