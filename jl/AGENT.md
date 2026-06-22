# Pluto Research Notebook — Agent Guide

This document contains everything an agent needs to build a new interactive research applet in this directory that matches the established conventions exactly. Read it fully before writing a single line.

---

## What these notebooks are

Each subdirectory under `jl/` is a self-contained research applet: one `app.pluto.jl` file that a viewer can open in Pluto and get a fully interactive, animated, publication-quality experience. The notebook:

- Opens with **all code hidden** — only outputs (animations, controls, metrics, takeaways) are visible
- Has a **full-width dark theme** at 1480 px max, matching a laptop screen without horizontal scrolling
- Contains a **live WGLMakie animation** driven by `Observable`s (no page reload needed)
- Has a **unified control panel** (`PlutoUI.combine`) so the viewer can change parameters and see every panel update reactively
- Closes with a **"What I take away from this"** section explaining the science in sincere student language
- Is entirely self-contained: all functions defined inline, embedded `PLUTO_PROJECT_TOML_CONTENTS` and `PLUTO_MANIFEST_TOML_CONTENTS`

---

## The template

A shared Julia module lives at `jl/template/PlutoTemplate.jl`. It provides CSS, header, section dividers, the takeaways layout, and the WGLMakie dark theme — all consistent across projects.

### Two-line bootstrap (goes in the imports cell)

```julia
include(joinpath(@__DIR__, "..", "template", "PlutoTemplate.jl"))
using .PlutoTemplate
```

`@__DIR__` resolves to the directory of the notebook file, so the relative path always works regardless of where Pluto is launched from.

### What the template exports

| Export | Returns | Usage |
|--------|---------|-------|
| `notebook_css(; accent)` | `HTML(...)` | CSS cell — accent is `ACCENT_GREEN` or `ACCENT_VIOLET` |
| `notebook_header(title, subtitle; breadcrumb, tags)` | `@htl` result | header cell |
| `section_label(text)` | `@htl` result | divider cells between major sections |
| `takeaways(items)` | `@htl` result | closing essay cell |
| `makie_theme()` | `WGLMakie.Theme` | called as `set_theme!(PlutoTemplate.makie_theme())` before `Figure(...)` |
| `ACCENT_GREEN`, `ACCENT_VIOLET`, `ACCENT_AMBER` | `String` (hex) | accent color selection |
| `MK_BG`, `MK_SF`, `MK_BD`, `MK_T0`–`MK_T2` | `RGBf` | background, surface, border, text levels |
| `MK_BLUE`, `MK_GREEN`, `MK_AMBER`, `MK_VIOLET` | `RGBf` | per-project data accent colors |

**Accent convention:**
- `ACCENT_GREEN` + `MK_BLUE`/`MK_GREEN`/`MK_AMBER` — physics, mathematics, simulation
- `ACCENT_VIOLET` + `MK_VIOLET`/`MK_GREEN` — information theory, complex systems, causal models

---

## Exact cell order (required)

Every notebook must follow this order. Every cell is `╟─` (code folded) in the cell order section — **no exceptions**. The viewer should open the notebook and see only outputs.

```
╟─ imports         using + DifferentialEquations etc. + include(...PlutoTemplate.jl) + using .PlutoTemplate
╟─ css             PlutoTemplate.notebook_css(; accent = PlutoTemplate.ACCENT_GREEN)
╟─ header          PlutoTemplate.notebook_header("Title", "subtitle"; breadcrumb=..., tags=[...])
╟─ library         begin ... end  — all pure Julia functions (data gen, algorithms, helpers)
╟─ section:ctrl    PlutoTemplate.section_label("Controls")
╟─ controls        @bind ctrl PlutoUI.combine() do Child  @htl """..."""  end
╟─ pipeline...     data → computation → derived quantities (one begin...end each, folded)
╟─ metrics         let  ... @htl """..."""  end   — stats dashboard
╟─ section:anim    PlutoTemplate.section_label("Animation — ...")
╟─ animation       begin  ... Observable + @lift + @async + fig  end
╟─ takeaways       PlutoTemplate.takeaways([("heading", "body html"), ...])
╟─ PLUTO_PROJECT_TOML_CONTENTS
╟─ PLUTO_MANIFEST_TOML_CONTENTS
```

---

## The fundamental HTML rule

> **`html"""$(x)"""` never evaluates `$(x)`.** It is `Base.Docs.@html_str`, which creates an `HTML{String}` from the raw string before interpolation runs. The `$(x)` is stored literally and rendered as the characters `$(x)` in the browser.
>
> **Always use `@htl """..."""`** (HypertextLiteral) for any cell that embeds Julia values into HTML output.

The only correct use of `html"""..."""` (no interpolation needed):
- The CSS cell — but the template handles this for you via `notebook_css()` which uses regular Julia string interpolation into a plain `HTML(string)` constructor, which is different from the `@html_str` macro.

---

## Controls — `PlutoUI.combine` with `@htl`

All interactive parameters are bound through **one** `@bind ctrl PlutoUI.combine()` cell that returns a `NamedTuple`. Downstream cells access `ctrl.fieldname`.

```julia
@bind ctrl PlutoUI.combine() do Child
    @htl """
    <div style="background:#111113;border:1px solid #222226;border-radius:10px;
                padding:24px 28px;display:flex;flex-direction:column;gap:22px;
                font-family:ui-sans-serif,system-ui,sans-serif;margin:8px 0 4px;">

      <!-- ── System selector ── -->
      <div style="display:grid;grid-template-columns:1fr auto;align-items:start;gap:20px;">
        <div>
          <div style="font-size:10px;color:#52525b;letter-spacing:.1em;text-transform:uppercase;font-weight:600;margin-bottom:4px;">Parameter Name</div>
          <div style="font-size:11px;color:#52525b;line-height:1.5;">Plain-language explanation of what this control does.</div>
        </div>
        <div style="min-width:200px;">
          $(Child("name", Select(["a"=>"Option A", "b"=>"Option B"]; default="a")))
        </div>
      </div>

      <div style="border-top:1px solid #1e1e22;"></div>

      <!-- ── Sliders ── -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
        <div>
          <div style="font-size:12px;color:#a1a1aa;font-weight:500;margin-bottom:2px;">Slider label</div>
          <div style="font-size:11px;color:#52525b;margin-bottom:8px;line-height:1.45;">What it controls physically.</div>
          $(Child("param1", Slider(0.1:0.1:5.0; default=1.0, show_value=true)))
        </div>
        <div>
          <div style="font-size:12px;color:#a1a1aa;font-weight:500;margin-bottom:2px;">Another slider</div>
          <div style="font-size:11px;color:#52525b;margin-bottom:8px;line-height:1.45;">What it controls physically.</div>
          $(Child("param2", Slider(1:1:10; default=3, show_value=true)))
        </div>
      </div>

    </div>
    """
end
```

**Why `@htl` is mandatory inside `combine`:** `PlutoUI.combine` uses `RenderCallback` objects returned by `Child()`. HypertextLiteral's `@htl` renders them lazily by calling `show(MIME"text/html"(), rc)`, which fires the callback that registers each bond. `html"""..."""` evaluates `$(Child(...))` eagerly to a raw string, the callback never fires, `captured_bonds` stays empty, and `ctrl` initialises as `missing` — crashing every downstream cell.

**Downstream usage:**
```julia
ctrl.name          # String from Select
Float64(ctrl.param1)   # coerce if needed (Slider returns Float64 already)
Int(ctrl.param2)       # coerce Int sliders
parse(Float64, ctrl.alpha_str)  # when Select returns a string like "1e-5"
```

---

## Animation — two patterns

### Pattern A: continuous physics (frame index + `@lift`)

Use when the system evolves continuously in time (ODEs, pendulums, orbits).

```julia
begin
    _n    = length(times)
    _STEP = Int(ctrl.speed)
    _TRAIL = Int(ctrl.trail)

    _t = Observable(1)   # frame index, 1..n

    # Lifted quantities update automatically when _t changes
    _pos_x = @lift [states[$_t, 1]]
    _pos_y = @lift [states[$_t, 2]]
    _trail_x = @lift states[max(1, $_t - _TRAIL):$_t, 1]
    _trail_y = @lift states[max(1, $_t - _TRAIL):$_t, 2]

    set_theme!(PlutoTemplate.makie_theme())
    fig = Figure(size = (1440, 720))

    ax = Axis(fig[1,1]; title = "Phase space")
    lines!(ax, _trail_x, _trail_y; color = (PlutoTemplate.MK_BLUE, 0.7), linewidth = 1.8)
    scatter!(ax, _pos_x, _pos_y; color = PlutoTemplate.MK_AMBER, markersize = 14)

    set_theme!()

    @async try
        while true
            _t[] = mod1(_t[] + _STEP, _n)
            sleep(0.033)    # ~30 fps
        end
    catch
    end

    fig
end
```

### Pattern B: stepped discrete systems (matrix + `notify`)

Use when the system evolves in discrete steps (cellular automata, agent models).

```julia
begin
    _T = 200; _W = 200; _DT = 0.045

    _matrix = fill(NaN32, _W, _T)
    _obs    = Observable(_matrix)

    set_theme!(PlutoTemplate.makie_theme())
    fig = Figure(size = (1440, 760))

    ax = Axis(fig[1,1]; title = "Space-time diagram")
    heatmap!(ax, _obs; colormap = [:gray10, :gray95],
             nan_color = PlutoTemplate.MK_SF, colorrange = (0f0, 1f0))

    set_theme!()

    @async try
        while true
            fill!(_obs.val, NaN32)
            notify(_obs); sleep(0.2)
            for τ in 1:_T
                _obs.val[:, τ] .= Float32.(data[τ, :])
                notify(_obs); sleep(_DT)
            end
            sleep(1.5)
        end
    catch
    end

    fig
end
```

**Pattern A vs B decision:** if you have a pre-computed trajectory array, use A. If you are replaying a discrete simulation step-by-step (and want the heatmap to "fill in" progressively), use B.

---

## Metrics dashboard

```julia
let
    # Pre-compute all display values before the @htl block
    val1 = round(some_float; sigdigits = 3)
    val2 = some_string
    color1 = some_condition ? "#22c55e" : "#f97316"

    @htl """
    <div style="background:#111113;border:1px solid #222226;border-radius:8px;
                overflow:hidden;margin:4px 0 8px;font-family:ui-sans-serif,system-ui,sans-serif;
                display:grid;grid-template-columns:repeat(N,1fr);">
      <div style="padding:14px 18px;border-right:1px solid #222226;">
        <div style="font-size:10px;color:#52525b;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">Label</div>
        <div style="font-size:14px;color:$(color1);font-weight:500;font-family:ui-monospace,monospace;">$(val1)</div>
      </div>
      <div style="padding:14px 18px;">
        <div style="font-size:10px;color:#52525b;letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;">Label 2</div>
        <div style="font-size:14px;color:#e4e4e7;font-weight:500;">$(val2)</div>
      </div>
    </div>
    """
end
```

**Rule:** always pre-compute display values into local variables (`val1`, `val2`, etc.) before the `@htl` block. Do not call functions inside `$(...)` expressions — keep interpolations simple variable references. This prevents silent rendering errors.

---

## Takeaways section

```julia
PlutoTemplate.takeaways([
    ("What the system is showing",
     "Body text with HTML &mdash; entities and <em>inline tags</em> are fine.
      Write as a sincere student explaining what you genuinely understood."),

    ("What the algorithm does",
     "Explain the core method without jargon overload.
      Reference the controls: <em>tune the &gamma; slider until Panel D looks like Panel C</em>."),

    ("What each parameter controls",
     "Walk through the physical meaning of each slider.
      Connect slider motion to visible changes in the animation."),

    ("Honest limitations",
     "What assumptions were made. Where the method breaks. What you would do next.
      This is the most important section &mdash; intellectual honesty matters."),
])
```

The body string is passed through as raw HTML. Include HTML entities and inline tags freely. Write in first-person student voice: "I noticed...", "What surprised me was...", "The part I'm still unsure about is...".

---

## Package conventions

### Always in `PLUTO_PROJECT_TOML_CONTENTS`

```toml
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
HypertextLiteral = "~1.0.0"
PlutoUI = "~0.7.80"
WGLMakie = "~0.13.9"
```

### Add per-project

| Need | Package | UUID |
|------|---------|------|
| ODEs / differential equations | `DifferentialEquations` | `0c46a032-eb83-5123-abaf-570d42b7fbaa` |
| Matrix algebra | `LinearAlgebra` | `37e2e46d-f89d-539d-b4ee-838fcccc9c8e` |
| Statistics | `Statistics` | `10745b16-79ce-11e8-11f9-7d13ad32a3b2` |
| Random numbers | `Random` | `9a3f8284-a2c9-5f02-9a11-845980a1fd5c` |

### DO NOT add

- `AbstractPlutoDingetjes` — transitive dep of PlutoUI, already in manifest
- `Colors`, `ColorTypes` — re-exported by WGLMakie as `RGBf`, `RGBAf`, etc.

---

## New project checklist

1. **Create directory** `jl/my_project/`
2. **Copy** an existing `app.pluto.jl` as starting point (prefer `conservation_laws/app.pluto.jl` for continuous physics, `causal_emergence/app.pluto.jl` for discrete/information-theory)
3. **Pick accent color** — `ACCENT_GREEN` for physics/math, `ACCENT_VIOLET` for info-theory/complex-systems
4. **Write the library cell** — all pure functions, no UI, ends with `nothing`
5. **Build the combine control panel** using `@htl` — one `ctrl` NamedTuple, named fields with plain-language labels and descriptions
6. **Write pipeline cells** — one `begin...end` per logical step, each assigns to a variable that downstream cells depend on
7. **Write metrics cell** using `@htl` — pre-compute display values before the template
8. **Write animation cell** — pick Pattern A or B, use `PlutoTemplate.makie_theme()` and `MK_*` constants
9. **Write takeaways** — minimum four sections: concept, method, parameters, limitations
10. **Set all cells to `╟─`** in the cell order section
11. **Update `PLUTO_PROJECT_TOML_CONTENTS`** with all required deps and their UUIDs from the manifest
12. **Verify** by running `julia --project=. -e 'using Pkg; Pkg.instantiate()'` in the project directory

---

## Critical pitfalls (learned the hard way)

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| `html"""$(x)"""` to interpolate values | Renders literal `$(x)` text | Use `@htl """$(x)"""` |
| `html"""..."""` inside `combine` do-block | `captured_bonds` = 0, `ctrl = missing`, every downstream cell crashes | Use `@htl """..."""` inside combine |
| `Float64(ctrl.alpha)` when `alpha` is a `Select`-returned `String` | `MethodError: no method matching Float64(::String)` | Use `parse(Float64, ctrl.alpha)` |
| Calling complex expressions inside `@htl` `$(...)` | Silent render errors or unexpected output | Pre-compute into local vars first |
| Cells marked `╠═` in cell order | Code editor visible on notebook open | All cells must be `╟─` |
| `using HypertextLiteral` in TOML but not in `using` line | `@htl` undefined in cells | Add to both the `using` line AND the TOML |
| Changing PLUTO_PROJECT_TOML_CONTENTS without including HypertextLiteral | Pluto fails to load `@htl` | Always include HypertextLiteral in TOML |

---

## Directory structure reference

```
jl/
├── template/
│   └── PlutoTemplate.jl          ← shared module (this is the LaTeX template)
├── AGENT.md                       ← this file
├── conservation_laws/
│   └── app.pluto.jl               ← Pattern A example (continuous ODE physics)
├── causal_emergence/
│   └── app.pluto.jl               ← Pattern B example (discrete cellular automaton)
└── <new_project>/
    └── app.pluto.jl               ← your new notebook
```
