# eval/locked

Real, hand-labelled test sets live here, one folder per project. Everything in this
folder except this README is git-ignored.

- Only humans add or change files here.
- Only eval harnesses read them, during nightly runs.
- Never use these files for training, few-shot examples, prompts, thresholds or error
  analysis that feeds back into training. A number measured on a set you have tuned
  against is not a result.
