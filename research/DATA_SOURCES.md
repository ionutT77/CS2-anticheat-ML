# Data and attribution

## CS2CD

The CS2 experiment uses [CS2CD.Counter-Strike_2_Cheat_Detection](https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection), published by CS2CD under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Associated work: Mille Mei Zhen Loo, Gert Lužkov and Paolo Burelli, *AntiCheatPT: A Transformer-Based Approach to Cheat Detection in Competitive Computer Games* (2025), [arXiv:2508.06348](https://arxiv.org/abs/2508.06348).

Pinned raw-data revision: `44e5129654508b22802a050a45bcdbb44b103d87`.
The source Git/LFS digests are in `data/cs2cd/metadata/raw_files.json`.
The completed download checks are in `raw_verified.jsonl` in that directory.

The included validation example, event index, player tables and derived predictions come from this release. Our processing extracts 256-tick windows before bullet damage, derives 39 channels, creates match groups and trains new classifiers. This changes the authors' window and evaluation design. These scores are not a reproduction of the paper's benchmark. The example contains one match-local player from validation match `no_cheater_present/1`; it was not drawn from the reserved test set.

## Kaggle CS:GO dataset

The original prototype and LSTM repair use [CSGO cheating dataset](https://www.kaggle.com/datasets/emstatsl/csgo-cheating-dataset), published by emstatsl, version 1. The publisher lists its license as Unknown. Raw samples are excluded from this handoff; retrieve them from the publisher for local use.

Archive SHA-256:
`09cc914e7201f65cfb38972fb6398199d1ac266bf0c8deea24128cf7b6d3f0db`.

The authoritative digest and file sizes are in `data/raw/download_manifest.json`.
The archive contains 10,000 negative-label and 2,000 positive-label records, each with 30 engagements. A record is not a verified unique person: the release lacks stable player and match identifiers.

## Original prototype and graphics

The original code and journal belong to the [CS2-anticheat-ML project](https://github.com/ionutT77/CS2-anticheat-ML), reviewed at commit `06427ceabd9990a8bc278b21758e1b04aa9d46f7`. No repository-wide software license was present at that revision. This handoff does not assign a new license to the upstream work.

The telemetry illustration combines an official Valve gameplay screenshot, an original aim-angle drawing and traces from the included CS2CD validation example. The screenshot and traces depict different encounters. See [illustration sources and rights](../docs/assets/CREDITS.md). The results figure is generated from the saved experiment metrics.
