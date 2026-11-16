# Prompt: deep analysis and enhancement roadmap for the CS2 anti-cheat thesis

Paste everything below the line into a new Claude Code session opened at the repository root.

---

I'm the author of this repository, a bachelor's thesis titled "Machine Learning-Based Server-Side Anti-Cheat for Counter-Strike 2". I want you to analyze the whole project, find its real weaknesses (especially in the data), and work out with me how to take it to the next level. I've been deep in it for months and have reached a point where several decisions are open at once, so what I need most is an independent, critical read from someone who checks the evidence rather than trusting my notes, followed by concrete proposals backed by numbers where numbers are obtainable.

## What the project is

The model scores one player in one match ("player-match") for how likely their aim behaviour is cheating. Inputs are four-second encounter windows (256 ticks at 64 Hz, 39 channels) ending strictly before the first bullet hit. The selected model is a blend of a temporal CNN and LightGBM with Platt calibration, trained and evaluated on the CS2CD dataset with complete-match separation between splits. There is also a working real-time pipeline (a server plugin, a window builder and live scoring), which so far has only been run by me against bots. The output is a calibrated score and review flags, never an automatic ban.

## Where to start

A survey of the repository was written on 2026-11-16 at `research/notes/project-state-2026-11-16.md`. Read it first, in full. It covers the datasets, the pipeline, the reported metrics, known data issues, open decisions, gaps, and an index of the important files, and it tags every claim as read, checked, inferred or unverified.

Treat that note as a map and not as ground truth. It was produced by a read-only pass that ran no code, and its last section lists what it could not verify. Before you build an argument on any number from it, or from my own documents, open the artifact it cites and confirm it. Where the documents disagree, `PROJECT_JOURNAL.md` is the most current record; `README.md` and `research/ARCHITECTURE_CHOICES.md` are reported to be stale about the scout work and the live pipeline. The project's `.claude/CLAUDE.md` lists the invariants the work must respect (no post-hit leakage, match-separated splits, no threshold tuning on test, and so on); they apply to everything you propose or run.

## What I want examined

The survey surfaced several things that could undermine the headline result. I'd like each one investigated as a hypothesis, with evidence for and against, and a statement of how much of the reported performance it could explain:

- Every positive label comes from one source folder (`with_cheater_present`), and a source-only baseline is reported to reach a high AUC. How much of the model's signal is "which folder" rather than "how the player aims"?
- Cheating scout encounters heavily outnumber legitimate ones in training, while the AWP is not imbalanced. Most strict-threshold detections, and all strict false positives, are reported to be scout-heavy players. Is the model partly a scout-style detector?
- The strict false positives are reported to cluster in a single match from the unreviewed negative source. Is that label noise (unlabeled cheaters), a match-level artifact, or a real failure?
- The calibration split has few negatives, so very low false-positive-rate targets cannot be resolved. What thresholds can the data actually support, and with what uncertainty?
- Player IDs are remapped per match, so the splits separate matches but cannot be shown to separate people. How large could same-person leakage be, and is there any way to bound it?
- Negative labels are unreviewed and may be noisy. What does that do to the reported precision and false-positive rate?
- Pro-player demos (OpenCS2) are staged as legitimate-only data and were scored once; lenient thresholds reportedly flag a noticeable share of pro AWPers. What does this say about domain shift between pros, the CS2CD population and ordinary players?
- Some local demos were scored without any recorded ground truth. Decide what, if anything, they can be used for.

Beyond these, look for problems I have not noticed. I would rather hear about a flaw now than from the thesis committee.

## Open decisions

Four decisions are open, and I want to make them with you rather than have them made for me:

1. Friend data. I can record sessions with friends on my own server. One document fixes these as evaluation-only; later notes reopen whether to train on them, evaluate on them, or split by person. My worry is that encounters from the same player share a style, so they cannot sit on both sides of a split, and no strategy is written down yet.
2. Scout imbalance. Options noted so far include reweighting, a weapon-conditioned model and per-weapon calibration (the last was set aside for sample size).
3. OpenCS2 pro data. Whether to use it for training, keep it as a held-out legitimate set, or both with a split by player.
4. Per-weapon expert models versus retraining one model on everything. This is only mentioned in a commit message and has not been analyzed anywhere.
5. What about obtaining non-legitimate data, by using cheats in a private environment? This would help us obtain any type of data at any FOV of the cheat(silent or trigger aim).

For each one, first gather the evidence that bears on it (from the repo, from cheap experiments, from the literature), then ask me a focused question that lays out the options, what each costs, what the evidence says, and which you would choose. Ask about one decision at a time when you have something concrete to show for it, and keep working on whatever does not depend on my answer while you wait. Ask me for the thesis deadline at the same point, since it affects what is feasible, and for anything else only I can know, such as how many friends could realistically record sessions.

## Brainstorming enhancements

Once you understand the weaknesses, propose ways to make the project meaningfully better. My priorities, in order, are detection performance (better recall at low false-positive rates, more weapons and more cheat types than the current aim-focused scope; bunny-hop detection was an original goal and never started) and real-world deployability (the real-time detector working on independently collected data, towards a tool a server admin could actually use). Thesis polish matters less than those two, though anything that would make the results indefensible matters a great deal.

Search the literature and public sources for ideas: published work on FPS cheat detection, learning with noisy or positive-unlabeled labels, shortcut and confound mitigation, domain adaptation, calibration with few negatives, and other public CS2 or CS:GO datasets with trustworthy labels. Cite every external claim with a link and keep external findings visibly separate from what you found in the repository. Do not put SteamIDs, player nicknames, my friends' names or my own details into search queries or into anything you write.

For each idea, say what problem it addresses, what it would take (data, compute, time), what could go wrong, and how we would know it worked. Rank the ideas by expected impact on the two priorities per unit of effort. I'd rather have six ideas you would stake something on than twenty listed for completeness, and I want to know which popular-sounding ideas you considered and rejected, with the reason.

## Experiments

You have my approval to run experiments on this machine without asking each time, within these limits. This overrides the general rule in my global instructions about waiting for approval before training, for the cases described here only.

This machine has no CUDA GPU, and the CS2CD raw data and window cache are not stored locally, so the neural model cannot be retrained here. What you can run locally is CPU work on data already on disk: profiling the saved tables, re-scoring and re-analysing saved predictions, calibration and threshold studies, bootstrap intervals, LightGBM and other tabular baselines, ablations by weapon and by source, and the existing tests. Check what is actually on disk before planning around it. Keep individual runs to roughly fifteen minutes; if something needs longer, tell me what it is and why before starting it.

For anything that needs a GPU or the full dataset, write a ready-to-run Kaggle script and a short spec instead (what it trains, on which split, with which seed and config, what it outputs, and what result would confirm or refute the idea), following the pattern of the existing Kaggle scripts in the repo. I will launch those myself and bring the outputs back.

Still ask before downloading any data, installing dependencies, deleting or overwriting existing files, or changing existing pipeline code. Put new scripts and outputs in new files under `research/experiments/` in a clearly named subfolder, so nothing existing is altered.

Evaluation discipline matters more here than usual, because the CS2CD test split and the pro AWP set have both already been used for evaluation. Make every selection decision (features, hyperparameters, thresholds, weights) on training, validation or calibration data only. If you look at the test split, it is to report a pre-decided analysis once, and you should say so. For every experiment record the dataset version, split, seed, configuration, code revision, metrics and where the output is saved. Report precision, recall, false-positive rate and the threshold with how it was chosen, with uncertainty intervals where the sample is small; accuracy alone tells me nothing.

## What to hand back

Write the final report to `research/notes/thesis-analysis-report.md`. I'll read it cold, possibly days later, so lead with the conclusions: the three or four findings that most change how the project should be understood, then the detail. It should contain:

- The verified state of the project, noting anywhere the survey note or my documents turned out to be wrong.
- Each data issue, with the evidence, how serious it is, and how confident you are.
- Results of the experiments you ran, with their records, and the Kaggle specs you wrote.
- The four decisions, with what I chose and why, or your recommendation where I have not answered yet.
- The ranked enhancement roadmap, with a concrete ordered plan for the next steps.
- What you could not check, and what remains uncertain.

Keep three kinds of statement clearly distinguishable throughout: what you verified yourself, what you inferred, and what comes from an external source. Do not describe anything as validated live anti-cheat performance; the project has no independently collected deployment data yet, and the report should say plainly what would be needed to earn that claim. If the evidence shows the headline result is weaker than I have been reporting, say that directly and show me the numbers.

Do not commit anything; I'll review the files and commit myself.
