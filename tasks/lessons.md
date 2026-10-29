# Lessons

- 2026-10-18: Qwen3 chat template defaults to thinking ON. The z-lab Qwen3-4B/8B DFlash
  drafters are trained on NON-thinking responses, so always pass `--reasoning off`
  when benchmarking them, or acceptance length collapses (3.06 vs expected ~6).
  General rule: before measuring a drafter, confirm the decoding mode matches its training data.

- 2026-10-18 (user correction): This project is for Sasi to LEARN, not for me to produce results.
  I ran Module 0 end-to-end and declared it done; the user wanted to be taught step by step.
  Rules from now on:
  * I explain one concept at a time, with an analogy/picture, at beginner level.
  * The user runs the commands and writes the code; I give the exercise, they predict the
    outcome first, then run, then we compare. I only run things to demo or to debug.
  * Never mark a module "done" myself. Done = the user can explain it back.
  * Frame the subject as inference engineering broadly; DFlash is an example, not the goal.

- 2026-10-19: The book at book/index.html is now the primary learning record (replaces
  LEARNING_LOG.md as the thing Sasi reads). After EVERY exercise: add the result to the
  chapter, the Lab notebook table (Appendix A), move the "now" marker, unlock the next
  chapter, republish the artifact. Keep figures to scale; copper = measured, steel = ceiling.

- 2026-10-19 (user caught an error): I wrote "9 tokens / (1, 9, 151936)" as a prediction and then
  repeated it as fact in chat and the book; the real count was 8. Rule: never state a number
  the user will measure as if it were already measured. Write "about N" or "you will see N"
  and let their output be the source of truth. Fix the book immediately when corrected.

- 2026-10-19: I declared "the model runs at 90% of ceiling, loss is around it" from ONE 20-rep
  burst measurement; the 200-rep sustained run was 2× slower and overturned it. Rule: never
  draw a conclusion in the book from a single measurement; require a sustained run (as long
  as the real workload) before interpreting. When wrong, keep the wrong result in the book
  with a "revised" note — the correction is part of the lesson.

- 2026-10-20: The "40% of roofline" on Oct 18/19 (my runs AND Sasi's) was machine state, not
  software: Oct 20 generate = 55 tok/s = 99% of ceiling. Rules: (1) always re-run the baseline
  in the same session as the comparison; (2) record power source / low-power mode / background
  load with every measurement; (3) sanity-check impossibility (188 ms/step was a /2 vs /20 bug).
  My Oct 18 RESULTS.md "both at ~40% of ceiling" conclusion is wrong and should be annotated.

- 2026-10-20: Sasi asked for the book to read as a shareable textbook, not a diary of attempts.
  Rule for the book: neutral authorial voice, principles first, measured numbers as evidence,
  mistakes folded into "case notes" and "measurement rules", no "pending/revised" scaffolding,
  forthcoming chapters shown as outlines. Lab notebook + reproduction appendix hold the raw record.

- 2026-10-25: Repo is public at github.com/sasidhar92/learn-infer; book served by GitHub Pages
  from docs/ at https://sasidhar92.github.io/learn-infer/. Routine after EVERY book update:
  (1) edit book/index.html, (2) python3 book/build.py, (3) republish artifact,
  (4) git add -A && git commit && git push origin main. Sasi asked that new chapters be committed.
