# Manual test checklist

Nothing in the GUI sections below has ever run on a machine with Qt — my
sandbox has no Qt runtime, so those are first-run territory. The command-line
items *have* been executed here. Report per number; "works" or the traceback is
equally useful.

Versions: PyRATE-TA `0.1.260730.dev5`, PyMORGAN `0.6.260730.dev4`.


## 0. Before anything

```powershell
cd "C:\Users\ricar\switchdrive\Ambizione UniGE\Scripts\PyRATE-TA"
uv pip install -e ".[gui,dev]"
uv run pymorgan-install-fonts       # do this once: fonts are per environment
uv run pytest                       # expect ~222 passed
uv run python scripts/run_checks.py # expect 18/18, GUI check may be skipped
```

If `pytest` fails here, stop and send the output — the rest will be noise.


## 1. Command line, no GUI

1. `uv run python examples/quick_fit.py --synthetic --taus 3 100 --irf 0.3`
   Expect lifetimes near 8 and 300, and four windows (scheme, species spectra,
   concentration profiles, residuals).
2. Same with `--weights`: the statistic says `reduced chi2`, near 1, not `SSR`.
3. A real file: `uv run python examples/quick_fit.py "path\to\scan.pdat" --taus 1 20 500`
4. Same file with `--tmin` and `--probe-range`; the summary reports the window.
5. `--taus 1 20 inf`: the third prints as `inf (non-decaying) (fixed)`, no
   uncertainty.
6. `--save`, then `import pyrate_ta as pr; print(pr.load_fit("...prfit").summary())`
7. A file with no `.pdatn` sibling plus `--weights`: a clean refusal, not a
   traceback.


## 2. GUI: loading and display

8. `uv run pyrate-ta-gui` starts; the citation block appears in the console.
8b. **The window fits your 13-inch screen**: it asks for 940x900 and is
    clipped to the available area and centred, so nothing sits under the menu
    bar or dock. Taller than wide — say if the proportions are still off.
    Drag it smaller — the left column gains a scroll bar rather than the window
    refusing to shrink. **View → Restore default window size** re-centres it.
9. **Load Data** opens a dataset (PDAT only — the format selector is gone).
    The **Data** lamp turns green; the **Noise** lamp turns green when a
    `.pdatn` sibling or single scans were found, red when not, and the weighting
    checkbox follows it.
10. The plot area is **one** figure with **one** toolbar: contour top-left and
    twice the size, kinetics to its right (rotated, delays vertical), spectra
    below, controls panel in the free corner.
11. No duplicated tick labels: no probe labels under the contour, no delay
    labels left of the kinetics.
12. The kinetics delay axis lines up with the contour's (same range, same
    symlog break).
13. Data are drawn as translucent points, not lines.
13b. The **embedded** panels keep their legends inside (three panels share one
    figure; there is no room beside an axis). **Pop-out** figures follow
    `[plot] legend_location` in PyMORGAN's `settings.toml`, now back to
    `"outside right"` — set it to `"best"` or any Matplotlib `loc` to move
    them inside there too.
13c. The figure fills the space: margins are now left 0.09, right 0.975,
    top 0.95, bottom 0.11 (all six in `[plots]` of `settings.toml`, along with
    the panel gaps). Change one and restart to confirm it takes effect.
14. Plot-controls panel: colour scale, contour count, limits — the contour
    redraws. The panel is about half as tall as before (two blocks side by
    side) and every control still works.
14b. **Plot kinetics** and **Plot Tr. Spectra** in *Additional plots and cuts*:
    both now prompt for the cuts, pre-filled with the crosshair position, and
    accept `1900, 1950`, `1900:5:1950` and `all`. *Kinetics used to fail
    outright* — it was handed a smoothing width the plotter did not accept.
14c. With the smoothing spin box non-zero, both pop-outs smooth. A negative
    width draws the raw trace faintly under the smoothed one.
15. No version label at the bottom; **Help → About PyRATE-TA** carries it.
16. **Clear Data** blanks everything without error.


## 3. GUI: the crosshair

17. A dashed crosshair sits on the contour, starting at the strongest signal.
    The kinetics panel shows *that* probe position and the spectra panel *that*
    delay — one trace each, not an arbitrary set.
18. Drag the vertical line: the status bar reports the cut while dragging, and
    both trace panels redraw when you release.
19. Drag the horizontal line. Then click away from both lines: the whole
    crosshair jumps there.
20. The reported values snap to measured ones (a real pixel and a real delay).


## 4. GUI: the model panel

21. The lifetime table reads **Tau | LB | UB | Fix?**, bound columns narrow.
22. Untick **Gaussian IRF**: the **Δ** row disappears, **t0** stays and is still
    used by the fit. Tick it again: Δ returns.
23. The fit controls (Show / Preview / FIT DATA / Reset / statistic / Copy taus)
    are in the left column under the IRF box, not in a strip above the plots.
24. **Define model...** is disabled for Parallel and Sequential, enabled for
    Target.
25. Parallel, Sequential and Target have three different colours.
25b. **Fit coherent artefact (IRF + derivatives)**, under the Gaussian IRF box:
    tick it and fit a dataset with a real time-zero artefact. Expect a visibly
    better residual map around t0 and a shortest lifetime that no longer
    absorbs it. The concentration profiles gain three extra traces (`IRF`,
    `dIRF/dt`, `d2IRF/dt2`); the species spectra do **not**.
25c. Untick **Gaussian IRF**: the artefact box disables itself and clears, with
    the reason in its tooltip (the basis is built from the IRF).
25d. Every control in the left column fits vertically without scrolling — say if
    any group still needs the scroll bar.
25e. **Fit options** has no *More options...* button any more (it opened
    nothing; those settings are in **View → Settings → Solver**).
25f. The algorithm combo starts on **Trust Region Reflective**. Picking
    *Levenberg-Marquardt* and fitting with the default bounds (tau >= 0) is
    refused with a message naming the reason — it cannot honour bounds.


## 5. GUI: fitting

26. Two lifetimes, **Gaussian IRF** on, press **Preview**: no optimisation, the
    spectra panel updates, the statistic fills, no error dialog. *(This raised
    `NotImplementedError: cannot remove artist` from the crosshair; fixed in
    dev11 — the crosshair is now dropped before the figure is rebuilt.)*
27. Press **FIT DATA** straight afterwards. *This is the sequence that failed
    before.* Expect a progress dialog on a slow fit, lifetimes written back into
    the table, uncertainties in their tooltips.
27b. When the fit finishes, **two windows open by themselves**: the species
    spectra and the concentration profiles. Preview must *not* open them. The
    DAS/EAS/SAS legend sits at the *best* spot inside the axis and can be
    **dragged**.
27c. The spectra button is named after the model: **Plot DAS** for Parallel,
    **Plot EAS** for Sequential, **Plot SAS** for Target — and it renames
    itself when you switch family, before any fit.
27d. **Plot Conc. Profile**: one trace per species, same colours as the
    spectra, symlog delay axis, species named after the scheme for a target
    fit. Legend inside.
27e. **Show kinetic graph** (next to *Define model...*): draws the scheme with
    $k_1, k_2, \dots$ symbols. Try it *before* fitting too — it should use the
    lifetimes as typed. Then with a **custom scheme**, before *and* after a fit:
    *the after case used to raise* `KeyError: unknown target scheme 'custom'`.
    Expect your own species names on the nodes. With Target selected but no
    scheme yet, expect an instruction, not a traceback.
27f. `[gui] plot_species_after_fit = false` and `plot_profiles_after_fit =
    false` in `settings.toml`: the windows stop opening themselves, the buttons
    still work.
27g. **Restrict fit to display limits** with a *narrowed spectral range*, fit,
    then **Plot SAS/EAS/DAS**. *This used to be a size mismatch* — the spectra
    cover the fitted window while the dataset carries the full probe axis.
27h. Legends read `320 fs` and `8 ± 0.2 ps`, not `320.95763496398745` and not
    `+/-`. `[axes] round_labels = false` and `[plot] label_digits = 6` in
    PyMORGAN's `settings.toml` give the long form back.
27i. The graphs are **horizontal** now (left to right, ground state below), in a
    wide figure rather than a tall column.
27j. Legends and the console summary quote `8.04 ± 0.02 ps`, not
    `8.036 ± 0.0234`: the error goes to one significant figure and the value to
    that same decimal place. `[plots] round_uncertainties = false` restores the
    full precision; `show_uncertainties = false` drops the ± entirely.
27k2. **Untick Fix? on t0 and/or Δ** and fit. *This used to raise "fixed mask
    has N entries, expected N+1".* Expect a normal fit, with the fitted t0 and
    IRF width in the summary.
27k3. Set `[gui] fit_monitor_every = 5` and fit: a window of bars appears, one
    per free parameter, values on top and names below, updating as it runs.
    Fixed parameters are greyed. `fit_monitor_every = 0` (default) never opens
    it; `= 1` updates every evaluation and is visibly slower.
27k. The console no longer repeats the method references on every fit (the
    start-up banner carries them), and each run is preceded by a `====` rule
    naming it.
28. **Show: Fit** and **Residuals**: the contour panel switches. Residuals
    should look structureless if the fit is good.
29. **Copy taus**, paste somewhere: lifetimes and uncertainties, tab-separated.
30. **Reset** returns to the data view and disables the fit-only buttons.
31. Tick **Restrict fit to display limits**, zoom via the plot controls, watch
    the status bar report the window, fit again; the summary records it.
32. Type `inf` in a lifetime row and fit: it stays fixed at `inf`.
33. Type `abc` in a lifetime row and fit: a warning naming the row, not a crash.
34. **Cancel** a long fit: "Fit cancelled", not "Fit failed".
35. Weighted fit with a `.pdatn`: checkbox enabled, the label beside the
    read-out changes from `SSR:` to `red. chi2:`.
36. Without noise: checkbox disabled, tooltip explains why.


## 6. GUI: the scheme editor

37. **Define model...** (with Target selected) opens the editor: default
    three-step scheme, graph, and the K matrix beside it.
38. Break the text (delete a `: k1`) and press **Update K graph**: an error
    naming the line, no graph, OK disabled.
39. Fix it: graph returns, OK enables.
40. Pick a template from the drop-down: text, graph and matrix all update.
41. Write a scheme with a **shared rate name**:
    ```
    S1 -> ICT : k_ct
    S1 ->     : k_d
    ICT ->    : k_d
    init S1 = 1
    ```
    Expect 3 species, **2** rate constants, both `k_d` arrows labelled the same.
42. Accept it: the rate table becomes one row per rate constant, named after
    them, and Target is selected.
43. Fit with that scheme, then **File → Save fit session** and **Open fit
    session**: the summary quotes the scheme.


## 7. Settings

44. `settings.toml` at the repo root has `[fit] [solver] [irf] [lda] [plots]
    [gui] [paths]` with comments.
45. `[plots] data_marker = "s"`, `fit_linewidth = 2.5`, restart: the panels
    follow.
46. `[gui] gui_theme = "Fusion Dark"`, restart: dark window, plots still
    readable.
47. `uv run pyrate-ta-settings` opens the editor; saving writes back.
48. Delete `settings.toml`, start the GUI: recreated with defaults.
49. **View → Settings...** opens the in-window dialog: two tabs, *Analysis
    (PyRATE-TA)* and *Aesthetics (PyMORGAN)*. Every section of `settings.toml`
    now has a page (fit, solver, IRF, LDA, plots, interface, paths).
50. Change something aesthetic (colourmap, label style): the embedded plots
    re-render immediately. **Close** without saving, restart: it is back.
51. **Save permanently**: the status bar names *both* files. Check that
    PyMORGAN's fields went into PyMORGAN's `settings.toml`, not PyRATE-TA's.
52. Set `profile = "poster"` (or a `font_scale`) in **PyMORGAN's**
    `settings.toml` and restart PyRATE-TA: the plots must follow. *PyRATE-TA used to
    read only its own file*, so PyMORGAN's aesthetics — including the font
    stack — were ignored.
53. The console says nothing about fonts. If it warns that Helvetica /
    TeX Gyre Heros are missing, run `uv run pymorgan-install-fonts` and restart;
    that warning is the reason plots would look unlike PyMORGAN's.
54. `[plots] show_uncertainties` and `round_uncertainties` appear in the
    settings dialog (Analysis → Plots) and take effect on the next fit display.


## 8. Known gaps (not bugs)

- ODE-defined models: button disabled.
- Fit options dialog: button disabled.
- Absolute species spectra (ground state, sign constraints): Phase 6, not built.
- Lifetime density analysis: Phase 7, not built.
- Target fits are not reordered by lifetime; parallel and sequential are.


## What I most expect to break

Items 10–14 and 17–20 (the single figure and the crosshair are new), 23 (the
fit controls moved), 27 (preview-then-fit), 34 (cancelling raises through the
optimiser) and 37–43 (the dialog has never been instantiated). The console
traceback is worth more than a description.
