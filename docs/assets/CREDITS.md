# Illustration sources

The README illustration combines an official game screenshot, an original explanatory drawing and a real telemetry example. The screenshot and traces depict different encounters. The drawing explains horizontal aim error; it is not an annotation or measurement of the screenshot.

## Gameplay image

`cs2-gameplay.jpg` is an unmodified, 1920 × 1080 screenshot from Valve’s [Counter-Strike 2 Steam page](https://store.steampowered.com/app/730/CounterStrike_2/), retrieved on 7 September 2026. Game imagery is © Valve Corporation and retains its original rights.

[Original image](https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/730/ss_ef98db5d5a4d877531a5567df082b0fb62d75c80.1920x1080.jpg)

SHA-256: `efa4a751b0f9998eedaed934649d0233663782c8fb45a1411af56f840f8cfd62`.

## Telemetry

The traces use `research/examples/example_validation_player.npz`, encounter index 2, the first included encounter with a shot before the first hit. They show the raw `delta_yaw`, `target_error_yaw` and `shot` channels. Angular values are in degrees. All 256 samples precede the first hit, from −4 seconds to −1/64 second. Each row has its own labelled scale; no smoothing is applied.

The example comes from CS2CD. See [data attribution](../../research/DATA_SOURCES.md) for its pinned source and CC BY 4.0 license. Neither the screenshot nor the displayed example is presented as evidence of cheating.

## Drawing and typography

`telemetry.source.svg` contains the editable layout. `../build_telemetry.py` embeds the original image and fonts, plots the saved sample and exports the PNG used in the README. The accompanying `telemetry.svg` is self-contained.

Newsreader and Archivo are distributed under the SIL Open Font License. Their licenses are included in `fonts/`.
