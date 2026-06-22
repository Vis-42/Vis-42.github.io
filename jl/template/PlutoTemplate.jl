"""
PlutoTemplate — shared styling and layout module for interactive Pluto research notebooks.

Usage in any notebook's imports cell:
    include(joinpath(@__DIR__, "..", "template", "PlutoTemplate.jl"))
    using .PlutoTemplate

Then call the exported functions in dedicated cells:
    notebook_css(; accent = ACCENT_GREEN)      # CSS cell
    notebook_header("Title", "subtitle"; ...)  # header cell
    section_label("Animation")                 # section divider cells
    takeaways([("Heading", "body html"), ...]) # closing essay cell
    set_theme!(makie_theme())                  # before Figure(...)

Design language (2026 redesign): a quiet, paper-like dark "research notebook".
Serif prose for reading, a grotesque for labels, monospace for data; a deep
ink ground with a single accent per notebook, hairline rules, and generous
measure. Signatures and export names are unchanged from the previous version,
so every notebook keeps working — only the look changes.
"""
module PlutoTemplate

using HypertextLiteral

# HTML = Base.Docs.HTML — needed for $(HTML(raw_string)) inside @htl to bypass escaping.
const HTML = Base.HTML

# ─── palette constants (used for WGLMakie themes and fallback values) ────────────
# names preserved for backward compatibility; values refreshed for the redesign.
const DARK_BG       = "#0f1117"   # deep ink ground
const SURFACE       = "#161a23"   # card
const BORDER        = "#232936"   # hairline
const BORDER_STRONG = "#2f3645"   # card border
const TEXT_BRIGHT   = "#eef0f5"
const TEXT_MID      = "#b3bac8"   # readable secondary
const TEXT_DIM      = "#8a92a3"
const TEXT_GHOST    = "#626b7c"   # labels (lifted for contrast)
const TEXT_MUTE     = "#3c4350"
const ACCENT_GREEN  = "#57c98a"
const ACCENT_VIOLET = "#a594f2"
const ACCENT_AMBER  = "#e6a356"
const SURFACE_RAISED= "#1c212c"
const SERIF         = "'Source Serif 4', 'Iowan Old Style', Georgia, 'Times New Roman', serif"
const SANS          = "'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif"
const MONO          = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace"

const _FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');"

# ─── notebook_css ──────────────────────────────────────────────────────────────
"""
    notebook_css(; accent = ACCENT_GREEN)

Return the full Pluto notebook CSS as an `HTML(...)` object. Dark, paper-like
research-notebook theme with serif prose. `accent` tints rules, links, headings
and the running-cell marker.
"""
function notebook_css(; accent::String = ACCENT_GREEN)
    HTML("""
    <style>
      $(_FONT_IMPORT)
      /* ── Theme variables ──────────────────────────────────────────────── */
      :root {
        --nb-bg:           $(DARK_BG);
        --nb-surface:      $(SURFACE);
        --nb-surface-2:    $(SURFACE_RAISED);
        --nb-border:       $(BORDER);
        --nb-border-str:   $(BORDER_STRONG);
        --nb-text:         $(TEXT_BRIGHT);
        --nb-text-mid:     $(TEXT_MID);
        --nb-text-dim:     $(TEXT_DIM);
        --nb-text-ghost:   $(TEXT_GHOST);
        --nb-text-mute:    $(TEXT_MUTE);
        --nb-accent:       $(accent);
        --nb-serif:        $(SERIF);
        --nb-sans:         $(SANS);
        --nb-mono:         $(MONO);
        --nb-cm-bg:        #131720;
        --nb-cm-gutter:    #131720;
        --nb-cm-gutter-fg: #3c4350;
        --nb-cm-text:      #d7dbe6;
        --nb-sel-bg:       #243a55;
        --nb-active-line:  rgba(255,255,255,0.028);
      }

      /* ── Layout ───────────────────────────────────────────────────────── */
      main { max-width: 1280px !important; margin: 0 auto !important; padding: 0 30px !important; }
      pluto-cell > pluto-output { width: 100% !important; }
      pluto-notebook { background: var(--nb-bg) !important; }
      body           { background: var(--nb-bg) !important; }
      pluto-cell, pluto-cell:hover, pluto-cell:focus-within {
          background: transparent !important;
          box-shadow: none !important;
      }
      pluto-cell.code_folded > pluto-input { display: none !important; }
      pluto-cell > pluto-input {
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
      }
      pluto-cell.running > pluto-output, pluto-cell.queued > pluto-output {
          border-left: 2px solid var(--nb-accent) !important;
      }
      pluto-cell > pluto-output { background: transparent !important; padding: 2px 0 !important; }

      /* ── Prose: serif body for reading ────────────────────────────────── */
      pluto-output { color: var(--nb-text-mid); font-family: var(--nb-serif); }
      pluto-output p, pluto-output li {
          font-family: var(--nb-serif) !important;
          font-size: 15.5px; line-height: 1.78; color: var(--nb-text-mid);
      }
      pluto-output a { color: var(--nb-accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }
      pluto-output h1, pluto-output h2, pluto-output h3 {
          font-family: var(--nb-serif) !important; color: var(--nb-text);
          font-weight: 600; letter-spacing: -0.01em; line-height: 1.2;
      }
      pluto-output h1 { font-size: 27px; margin: 0.6em 0 0.3em; }
      pluto-output h2 {
          font-size: 20px; margin: 1.4em 0 0.5em; padding-bottom: 0.3em;
          border-bottom: 1px solid var(--nb-border);
      }
      pluto-output h3 { font-size: 16px; color: var(--nb-text-mid); margin: 1.1em 0 0.4em; }
      pluto-output blockquote {
          border-left: 2px solid var(--nb-accent); margin: 1em 0; padding: 0.2em 0 0.2em 1em;
          color: var(--nb-text-dim); font-style: italic;
      }
      pluto-output code, pluto-output pre, pluto-output kbd {
          font-family: var(--nb-mono) !important; font-size: 0.86em;
      }
      pluto-output :not(pre) > code { color: var(--nb-text); background: var(--nb-surface);
          padding: 1px 5px; border-radius: 3px; border: 1px solid var(--nb-border); }
      pluto-output table { color: var(--nb-text-mid); font-family: var(--nb-sans); font-size: 13px; }
      pluto-output th { color: var(--nb-text); border-bottom: 1px solid var(--nb-border-str) !important; }
      pluto-output td, pluto-output th { border-color: var(--nb-border) !important; }

      /* ── Code editor ──────────────────────────────────────────────────── */
      .cm-editor, .cm-scroller { background: var(--nb-cm-bg) !important; border-radius: 6px; }
      .cm-gutters {
          background: var(--nb-cm-gutter) !important;
          border-right: 1px solid var(--nb-border) !important;
          color: var(--nb-cm-gutter-fg) !important;
      }
      .cm-content { color: var(--nb-cm-text) !important; font-family: var(--nb-mono) !important; }
      .cm-editor.cm-focused { outline: none !important; }
      .cm-selectionBackground, .cm-editor .cm-selectionBackground { background: var(--nb-sel-bg) !important; }
      .cm-activeLine { background: var(--nb-active-line) !important; }

      /* ── Figures stay within the page (no horizontal scroll) ──────────── */
      pluto-output img, pluto-output canvas, pluto-output svg,
      pluto-output .js-plotly-plot, pluto-output figure {
          max-width: 100% !important; height: auto !important;
      }
      pluto-notebook, body { overflow-x: hidden; }

      /* ── Sliders / widgets pick up the accent ─────────────────────────── */
      pluto-output input[type=range] { accent-color: var(--nb-accent); }

      /* ── Cell chrome ──────────────────────────────────────────────────── */
      .pluto-cell-controls { opacity: 0 !important; transition: opacity 0.15s; }
      .pluto-cell-controls:hover { opacity: 1 !important; }
      pluto-cell::before { display: none !important; }
      pluto-cell { margin: 3px 0 !important; }
    </style>
    """)
end

# ─── notebook_header ───────────────────────────────────────────────────────────
"""
    notebook_header(title, subtitle; breadcrumb = "Research", tags = String[], author = "Parth Bhargava")

Return the project masthead as a `HypertextLiteral.Result`.
"""
function notebook_header(title::String, subtitle::String;
                         breadcrumb::String = "Research",
                         tags::Vector{String} = String[],
                         author::String = "Parth Bhargava")
    tag_spans = [
        @htl """<span style="font-family:$(MONO);font-size:10.5px;color:var(--nb-text-dim);border:1px solid var(--nb-border-str);border-radius:3px;padding:3px 8px;letter-spacing:0.02em;">$(t)</span>"""
        for t in tags
    ]
    @htl """
    <div style="padding:40px 0 26px;border-bottom:2px solid var(--nb-border-str);position:relative;">
      <div style="font-family:$(MONO);font-size:10.5px;color:var(--nb-accent);letter-spacing:0.18em;text-transform:uppercase;margin-bottom:14px;">
        $(breadcrumb)
      </div>
      <h1 style="font-family:$(SERIF);margin:0 0 8px;font-size:34px;font-weight:600;color:var(--nb-text);letter-spacing:-0.015em;line-height:1.08;">
        $(title)
      </h1>
      <p style="font-family:$(SERIF);margin:0 0 16px;font-size:16px;color:var(--nb-text-dim);line-height:1.6;font-style:italic;">
        $(subtitle)
      </p>
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <span style="font-family:$(SANS);font-size:11.5px;color:var(--nb-text-ghost);letter-spacing:0.04em;font-weight:500;">$(author)</span>
        <span style="flex:0 0 auto;width:18px;height:1px;background:var(--nb-border-str);"></span>
        <div style="display:flex;gap:7px;flex-wrap:wrap;">$(tag_spans)</div>
      </div>
    </div>
    """
end

# ─── section_label ─────────────────────────────────────────────────────────────
"""
    section_label(text)

Return a section divider: an accent tick, an all-caps mono label, and a hairline
rule running to the right margin.
"""
function section_label(text::String)
    @htl """
    <div style="display:flex;align-items:center;gap:12px;padding:26px 0 12px;">
      <span style="flex:0 0 auto;width:3px;height:13px;background:var(--nb-accent);border-radius:2px;"></span>
      <span style="font-family:$(MONO);font-size:11px;color:var(--nb-text-mid);letter-spacing:0.16em;text-transform:uppercase;font-weight:500;white-space:nowrap;">$(text)</span>
      <span style="flex:1 1 auto;height:1px;background:var(--nb-border);"></span>
    </div>
    """
end

# ─── takeaways ─────────────────────────────────────────────────────────────────
"""
    takeaways(items::Vector{Tuple{String,String}})

Render the closing essay as a set of numbered take-aways.
"""
function takeaways(items::Vector{Tuple{String,String}})
    entries = join(["""
    <div style="display:flex;gap:16px;">
      <div style="flex:0 0 auto;font-family:$(MONO);font-size:13px;color:var(--nb-accent);font-weight:500;padding-top:2px;min-width:24px;">$(lpad(string(i),2,'0'))</div>
      <div>
        <div style="font-family:$(SANS);font-size:14px;color:var(--nb-text);font-weight:600;margin-bottom:6px;">$(heading)</div>
        <p style="font-family:$(SERIF);font-size:15px;color:var(--nb-text-dim);line-height:1.78;margin:0;">$(body)</p>
      </div>
    </div>""" for (i, (heading, body)) in enumerate(items)], "\n")

    HTML("""
    <div style="padding:38px 0 50px;border-top:2px solid var(--nb-border-str);margin-top:14px;">
      <div style="font-family:$(MONO);font-size:11px;color:var(--nb-accent);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:22px;">
        What I take away from this
      </div>
      <div style="display:flex;flex-direction:column;gap:24px;">
        $(entries)
      </div>
    </div>
    """)
end

# ─── explanation ───────────────────────────────────────────────────────────────
"""
    explanation(how::String, meaning::String)

Render the closing two-part essay. `how` is the methods section; `meaning` is
the theory section. Both may contain inline HTML.
"""
function explanation(how::String, meaning::String)
    block(label, text) = """
      <div style="margin-bottom:30px;">
        <div style="font-family:$(MONO);font-size:10.5px;color:var(--nb-text-ghost);letter-spacing:0.12em;text-transform:uppercase;font-weight:500;margin-bottom:12px;">
          $(label)
        </div>
        <p style="font-family:$(SERIF);font-size:15.5px;color:var(--nb-text-mid);line-height:1.85;margin:0;">$(text)</p>
      </div>"""
    HTML("""
    <div style="padding:38px 0 50px;border-top:2px solid var(--nb-border-str);margin-top:14px;">
      <div style="font-family:$(MONO);font-size:11px;color:var(--nb-accent);letter-spacing:0.16em;text-transform:uppercase;margin-bottom:24px;">
        Notes
      </div>
      $(block("How the method works", how))
      <div style="border-top:1px solid var(--nb-border);margin-bottom:30px;"></div>
      $(block("What it means, physically and mathematically", meaning))
    </div>
    """)
end

# ─── makie_theme ───────────────────────────────────────────────────────────────
"""
    makie_theme()

Return a WGLMakie `Theme` matching the notebook palette (deep ink ground,
hairline grid, accent-led color cycle).
"""
function makie_theme()
    # Grab whichever Makie backend the notebook loaded (CairoMakie or WGLMakie);
    # RGBf and Theme are re-exported by both, so the theme is backend-agnostic.
    parent = parentmodule(PlutoTemplate)
    Mk = isdefined(parent, :CairoMakie) ? getfield(parent, :CairoMakie) :
         isdefined(parent, :WGLMakie)   ? getfield(parent, :WGLMakie)   :
         getfield(parent, :Makie)
    RGBf  = Mk.RGBf
    Theme = Mk.Theme
    bg = RGBf(0.059, 0.066, 0.091)   # #0f1117
    sf = RGBf(0.086, 0.102, 0.137)   # #161a23
    bd = RGBf(0.184, 0.212, 0.271)   # #2f3645
    t0 = RGBf(0.933, 0.941, 0.961)
    t1 = RGBf(0.701, 0.729, 0.784)
    t2 = RGBf(0.541, 0.572, 0.639)
    Theme(
        backgroundcolor = bg,
        textcolor       = t0,
        fontsize        = 13,
        Axis = (
            backgroundcolor  = sf,
            titlecolor       = t0, titlesize  = 13, titlefont = :regular,
            xlabelcolor      = t1, ylabelcolor = t1,
            xticklabelcolor  = t2, yticklabelcolor = t2,
            spinecolor       = bd, spinewidth  = 0.8,
            xgridcolor       = (bd, 0.7), ygridcolor  = (bd, 0.7),
            xgridwidth       = 0.6,   ygridwidth  = 0.6,
            xtickcolor       = (t2, 0.5), ytickcolor  = (t2, 0.5),
        ),
        Axis3 = (
            titlecolor       = t0, titlesize  = 13,
            xlabelcolor      = t1, ylabelcolor = t1, zlabelcolor = t1,
            xticklabelcolor  = t2, yticklabelcolor = t2, zticklabelcolor = t2,
            xgridvisible     = true, ygridvisible = true, zgridvisible = true,
            xgridcolor       = (:white, 0.22),
            ygridcolor       = (:white, 0.22),
            zgridcolor       = (:white, 0.22),
            xgridwidth       = 0.7,  ygridwidth  = 0.7,  zgridwidth  = 0.7,
        ),
        Legend = (
            backgroundcolor = sf, labelcolor = t1,
            framecolor = bd, framewidth = 0.7, labelsize = 12,
        ),
    )
end

# ─── Exports ───────────────────────────────────────────────────────────────────
export notebook_css, notebook_header, section_label, takeaways, explanation, makie_theme
export DARK_BG, SURFACE, SURFACE_RAISED, BORDER, BORDER_STRONG
export TEXT_BRIGHT, TEXT_MID, TEXT_DIM, TEXT_GHOST, TEXT_MUTE
export ACCENT_GREEN, ACCENT_VIOLET, ACCENT_AMBER, SERIF, SANS, MONO

end # module PlutoTemplate
