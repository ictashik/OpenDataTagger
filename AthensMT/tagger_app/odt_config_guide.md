# OpenDataTagger (ODT) — Config File Guide for AI-Assisted Design

## Why you (the reader) are being given this file

A human is using **OpenDataTagger (Athena ODT)**, a tool that reads a CSV file
row-by-row and asks a local LLM (or, in image mode, a Stable Diffusion server)
to fill in new columns — one call per row per new column. What each new
column means, which existing columns feed it, and what output it should
produce is defined entirely by one small **config CSV file** that this
document teaches you how to write.

The human is about to show you a handful of real sample rows from their
actual dataset (not the whole file) and describe, in their own words, what
they want tagged, classified, extracted, scored, or generated from it. Your
job:

1. Understand their data from the sample rows (column names, value shapes,
   what a row represents).
2. Understand their goal from their description.
3. Design one or more **tags** (= output columns) that accomplish it, each
   with a well-written prompt.
4. Emit a ready-to-use **config CSV** — the exact file format ODT loads —
   plus a short plain-English explanation of what each tag does and why.

The config format is completely independent of which LLM model the human
ends up running it with (Ollama, any local model) — it is just a definition
of *what to ask, about which columns, under what conditions, per prompt*.
Nothing below requires you to know anything about ODT's internals beyond
this document.

---

## 1. The core idea

A config file is a CSV with **one row per output column ("tag")**. ODT
processes tags **in the order they appear in the file**, and for every row of
the user's data it walks the tag list top-to-bottom, calling the LLM once per
tag (unless a condition skips it — see §4). A tag can be told to see the
*already-generated* answers from earlier tags in the same row, which lets you
chain reasoning steps (e.g. "classify" → "if flagged, propose a fix").

Only two columns are required. Everything else is optional and can simply be
omitted from the CSV entirely if unused — ODT fills sensible defaults for any
missing column.

```csv
OutputColumn,PromptTemplate
Sentiment,"Classify the sentiment of this review as Positive, Neutral, or Negative: {review_text}"
```

That alone is a complete, valid, one-tag config.

---

## 2. Full field reference

| Column | Required | Type | Default if omitted | Meaning |
|---|---|---|---|---|
| `OutputColumn` | **Yes** | short string, no spaces recommended | — | Name of the new column this tag writes. Must be unique across the file. |
| `PromptTemplate` | **Yes** | free text, may be multi-line | — | The instruction sent to the LLM for this tag. See §3 for placeholders and §5 for how to phrase the *output rule* (yes/no, free text, category, number, etc.) — there is no separate "output type" field; it all lives here as plain instructions. |
| `ConditionField` | No | column name (a source column, or an earlier tag's `OutputColumn`) | `''` (no condition — always runs) | If set, this tag only runs when the named field's value satisfies `ConditionOp`/`ConditionValue`. |
| `ConditionOp` | No | one of exactly: `==`, `!=`, `contains`, `not_contains`, `is_empty`, `is_not_empty` | `==` | Comparison operator. Comparisons are case-insensitive string comparisons. **Any other value silently behaves as "always true"** — see §6. |
| `ConditionValue` | No | string | `''` | Right-hand side of the comparison. Ignored by `is_empty`/`is_not_empty`. |
| `DefaultValue` | No | string | `''` (shown to the user as `N/A`) | Value written to `OutputColumn` when the condition is false. **No LLM call happens in that case** — this is a free skip, not a fallback after a failed call. |
| `SendContext` | No | exactly the string `1` (anything else, including `true`/`yes`, means "off") | `''` (off) | Only meaningful together with `ConditionField` pointing at an earlier tag. When on, that earlier tag's own prompt, answer, and explanation are appended to this tag's prompt as a labeled context block — use this to let a "fix it" tag see *why* a "flag it" tag made its call. |
| `InputColumns` | No | comma-separated list of column names, **or** the literal sentinel `__NONE__` | `''` (falls through to the run's global context — see §3) | Restricts which columns are interpolated into *this tag's* prompt / shown to the LLM as context, overriding the global selection for this tag only. `__NONE__` means "show this tag literally zero row context" (distinct from leaving the field blank, which means "use the global default"). |
| `ImageParams` | No | JSON object (string) | `''` → `{}` | **Image-generation-mode projects only** — ignored for text tagging. See §8. |
| `RetrievalConfig` | No | JSON object (string): `{"enabled": true, "top_k": 3}` | `''` → `{"enabled": false, "top_k": 3}` | **Only meaningful if the project has a reference dataset attached and indexed in ODT** (a bulk CSV/TXT/MD/PDF the human uploads separately, outside this config file — see §9). Do not set `enabled: true` unless you know such a reference dataset exists; there's nothing to retrieve otherwise. |
| `NodeX`, `NodeY` | No | numeric string | `''` | Pure cosmetic position of this tag's box in ODT's visual canvas editor. **Never set these when generating a new config** — omit the columns entirely. They're irrelevant to behavior. |

Every column beyond `OutputColumn`/`PromptTemplate` is optional **per-file**,
not just per-row — if none of your tags need conditions, for instance, just
don't include `ConditionField`/`ConditionOp`/`ConditionValue` columns at all.

### What actually gets written to the output CSV at runtime (informational)

You don't need to create these yourself — ODT adds them automatically per
tag when it runs — but it's useful to know they exist:

- `<OutputColumn>` — the answer itself.
- `<OutputColumn>_exp` — a short explanation the LLM gives for its answer.
- `<OutputColumn>_sources` — only present if `RetrievalConfig.enabled` was true — which reference chunks were used.

---

## 3. How columns get into the prompt

Reference any column **by name**, wrapped in curly braces, directly inside
`PromptTemplate`:

```
Classify the urgency of this ticket: {subject} — {message}
```

At runtime, `{subject}` and `{message}` are replaced with that row's actual
values as plain text. This works for:

- any original column from the user's source CSV, and
- any earlier tag's `OutputColumn` in the same file (its generated answer, treated as a fact for later tags).

**Which columns are eligible to interpolate** is governed by two layers:

1. A **global context selection** the human makes once in ODT's UI (which
   source columns are "in scope" for the whole run) — this is not part of
   the config file you're writing; you don't need to set it, but keep your
   placeholders to columns you can reasonably expect them to have selected
   (usually: all the columns you were shown in the sample data).
2. Each tag's own `InputColumns`, which — only if you set it — narrows
   *that tag's* view to a specific subset (or, via `__NONE__`, to nothing).
   Leave it blank unless a tag genuinely needs a tighter or different view
   than the rest (e.g. a "propose a fix" tag that shouldn't see the
   original flagged value, only the corrected inputs).

---

## 4. Conditions and chaining (worked pattern)

A very common, powerful shape is: **one tag gates, a later tag acts.**

```csv
OutputColumn,PromptTemplate,ConditionField,ConditionOp,ConditionValue,DefaultValue,SendContext,InputColumns
needs_followup,"Does this support ticket require human follow-up? Reply with only the word YES or NO.

Subject: {subject}
Message: {message}",,,,,,
suggested_reply,"Draft a short, empathetic reply to this support ticket that a human agent can send with light edits.

Subject: {subject}
Message: {message}",needs_followup,==,YES,,1,"subject,message,needs_followup"
```

Reading this:
- `needs_followup` always runs (no `ConditionField`), and answers YES/NO.
- `suggested_reply` only runs when `needs_followup == YES` (case-insensitive).
  When it's `NO`, `suggested_reply` is set to `DefaultValue` (blank here, so
  it shows as `N/A`) and **no LLM call is made** for that row/tag — this
  is a deliberate cost/time saver, not an error path.
- `SendContext=1` means `suggested_reply`'s prompt also receives
  `needs_followup`'s own prompt + answer + explanation as extra context, so
  the model drafting the reply understands *why* it was flagged.
- `InputColumns` on `suggested_reply` narrows its view to exactly
  `subject, message, needs_followup` — even if the global context includes
  more columns (e.g. an internal customer ID it doesn't need to see).

You can chain more than two tags this way (gate → gate → act), as long as
each `ConditionField` points at a tag defined **earlier in the file**.

---

## 5. Designing the output rule (yes/no, category, number, free text, structured)

There is **no separate field** for "this tag's output type." ODT always
wraps every call the same way behind the scenes — it appends its own
`Best Answer: / Explanation:` instruction after your `PromptTemplate` and
parses the two back apart. **Do not include your own "Best Answer:" /
"Explanation:" instructions in `PromptTemplate` — ODT already adds them.**
Your prompt should only describe *what to decide*, not *how to format the
final envelope*.

The output rule is entirely something you specify in prose, inside the task
description. Patterns that work well:

- **Yes/No**: *"Reply with only the word YES or NO."*
- **Fixed category set**: *"Classify as exactly one of: Billing, Technical, Account, Other."*
- **Bounded number**: *"Return only an integer from 1 to 5. No units, no explanation in the answer itself."*
- **Free text**: *"Write a two-sentence summary in plain English."*
- **Structured-ish text** (still just a string column, but parseable): *"Return a comma-separated list of extracted keywords, lowercase, no more than 5."*

Be explicit and narrow ("only the word YES or NO", not "tell me if this is
urgent") — nothing downstream enforces or coerces the answer to match your
intended shape. If the LLM ignores the instruction, the raw text is written
as-is; there is no validation or retry on content shape.

---

## 6. Hard constraints and silent-failure pitfalls

These don't raise errors — they silently produce wrong or degraded results —
so get them right when you author a config:

- **Every `{Column}` you reference must be spelled exactly like a real
  source column name or an earlier tag's `OutputColumn`.** A typo leaves the
  literal text `{like_this}` in the prompt sent to the LLM instead of a
  value — no warning is raised.
- **`ConditionOp` must be exactly one of the six listed values.** Anything
  else (including a blank meant to mean "no comparison" — use a blank
  `ConditionField` for that instead) is silently treated as "condition met,
  tag always runs."
- **`ConditionField` must point at something already defined earlier in the
  tag list** (or an original source column). Pointing at a tag defined later
  in the file will silently read as empty, not as an error.
- **`SendContext` must be the literal string `1`**, not `true`, `yes`, or `True`.
- **CSV-escape anything with commas, quotes, or newlines inside
  `PromptTemplate`** — it's a normal CSV cell, so multi-line prompts must be
  wrapped in double quotes with internal quotes doubled (`""`), exactly like
  any other CSV field. If you're generating this file programmatically,
  use a real CSV writer rather than hand-joining strings with commas.
- **`OutputColumn` names must be unique** and are best kept simple
  (letters, numbers, underscores) since they become literal column headers
  in the tagged output file.

---

## 7. What ODT does with your finished config (for the human's benefit)

1. Save the CSV you produce as a plain `.csv` file.
2. In ODT, upload their source data CSV — optionally attaching this config
   CSV at the same time as the "Config File (Optional)" field — or upload
   the source CSV alone and paste/import the tags on the **Define Columns**
   page afterward.
3. On **Define Columns**, confirm the global input-column selection (§3),
   review each tag, then start the run.
4. ODT calls the LLM once per tag per row, top-to-bottom per §1, and writes
   a tagged copy of the CSV plus a log of every prompt/answer/explanation.

---

## 8. Image-generation mode (only if the project generates images, not text)

If — and only if — the human tells you their project generates an image per
row (Stable Diffusion) rather than a text/classification answer per row, a
tag's `PromptTemplate` becomes the image prompt (still supports the same
`{Column}` interpolation), and `ImageParams` may carry a JSON object with
any of these optional keys — all have sane defaults if omitted:

```json
{
  "model": "",
  "negative_prompt": "",
  "width": 512,
  "height": 512,
  "steps": 30,
  "guidance": 7.5,
  "seed": -1,
  "num_images": 1,
  "scheduler": "default",
  "loras": [{"id": "", "scale": 1.0}]
}
```

`ConditionField`/`Op`/`Value`/`DefaultValue`/`SendContext`/`InputColumns`
all work identically to text mode. `RetrievalConfig` does not apply in image
mode.

---

## 9. Retrieval-grounded tags (only if the human mentions a reference dataset)

If the human says they've attached a **bulk reference dataset** in ODT (a
canonical values table, a standards/spec document, etc. — this is uploaded
separately in the ODT UI, not part of this config file) and want a tag to
cross-check its answer against it, set that tag's `RetrievalConfig` to:

```json
{"enabled": true, "top_k": 3}
```

`top_k` is how many of the most relevant reference chunks get pulled into
that tag's prompt automatically. There is no `query_columns` setting —
retrieval always reuses whatever columns/context that same tag already has
access to (per its `InputColumns`) as the search query. Don't enable this
unless the human has confirmed a reference dataset is attached — there's
nothing to retrieve otherwise, and it will not error, just retrieve nothing
useful.

---

## 10. Checklist before you hand back a config

- [ ] Every tag has a non-empty `OutputColumn` (unique) and `PromptTemplate`.
- [ ] Every `{placeholder}` matches a real source column or an earlier tag's `OutputColumn`, spelled exactly.
- [ ] Every `ConditionField` (if used) refers to something defined earlier in the file.
- [ ] Every `ConditionOp` (if used) is one of the six exact allowed values.
- [ ] `PromptTemplate` clearly states the desired output *shape* (yes/no, category list, number range, free text) in plain English — you are not relying on any schema/type enforcement downstream.
- [ ] `PromptTemplate` does **not** itself ask for a "Best Answer:"/"Explanation:" wrapper — ODT adds that automatically.
- [ ] The file is valid CSV (proper quoting of multi-line/comma-containing prompts).
- [ ] `NodeX`/`NodeY` columns are omitted entirely.
- [ ] You've given the human a short plain-English explanation of what each tag does, in what order, and why — alongside the CSV.
