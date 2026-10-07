import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="Percolation on Simplicial Complexes")


@app.cell(hide_code=True)
def _():
    import json
    import marimo as mo
    import numpy as np
    return json, mo, np


@app.cell(hide_code=True)
def _(json, mo):
    import sys as _sys

    # The animation engine shared by every notebook is py/_shared/canvas.html;
    # the website serves the same file at /apps/_shared/canvas.html.
    if _sys.platform == "emscripten":
        from js import location
        from urllib.request import urlopen

        _engine = urlopen(f"{location.origin}/apps/_shared/canvas.html").read().decode()
    else:
        _engine = (mo.notebook_dir().parent / "_shared" / "canvas.html").read_text()

    def canvas_anim(spec, height="480px"):
        """Animate a spec (format: see the top of py/_shared/canvas.html)."""
        return mo.iframe(_engine.replace("__SPEC__", json.dumps(spec)), height=height)

    return (canvas_anim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Percolation on simplicial complexes

        Standard bond percolation on a graph exhibits a continuous (second-order) phase transition
        at a critical probability $p_c$: below $p_c$ only small isolated clusters exist, above it
        a giant component spans the system. When the unit of occupation is not a single edge but
        a higher-order simplex (a triangle or tetrahedron), the cooperative constraint changes the
        transition. This notebook sweeps $p$ from 0 to 1 for $k=1$ (edges), $k=2$ (triangles),
        and $k=3$ (tetrahedra) on a clique complex built from an Erdos-Renyi graph, and animates
        $S(p)$ and $\chi(p)$ filling in point by point.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    dim_sl  = mo.ui.slider(1, 3, value=1, step=1, label="dimension k", show_value=True)
    N_sl    = mo.ui.slider(20, 80, value=40, step=5, label="N (vertices)", show_value=True)
    pe_sl   = mo.ui.slider(0.05, 0.40, value=0.18, step=0.01, label="p_edge", show_value=True)
    rep_sl  = mo.ui.slider(3, 10, value=5, step=1, label="n_rep", show_value=True)
    controls = mo.hstack([dim_sl, N_sl, pe_sl, rep_sl], justify="start", gap=2)
    controls
    return controls, dim_sl, N_sl, pe_sl, rep_sl


@app.cell(hide_code=True)
def _(np, dim_sl, N_sl, pe_sl, rep_sl):
    dim_k  = int(dim_sl.value)
    N_vert = int(N_sl.value)
    p_edge = float(pe_sl.value)
    n_rep  = int(rep_sl.value)
    return dim_k, N_vert, p_edge, n_rep


@app.cell(hide_code=True)
def _(np):
    def gen_complex(N, p_e, seed=42):
        """
        Build a clique complex on N vertices using an Erdos-Renyi G(N, p_e) base graph.
        Returns arrays of edges, triangles, and tetrahedra (all sorted tuples).
        """
        _rng = np.random.default_rng(seed)
        _adj = np.zeros((N, N), dtype=bool)
        _edges = []
        for _i in range(N):
            for _j in range(_i + 1, N):
                if _rng.random() < p_e:
                    _adj[_i, _j] = _adj[_j, _i] = True
                    _edges.append((_i, _j))
        _triangles = []
        for (_i, _j) in _edges:
            for _k in range(_j + 1, N):
                if _adj[_i, _k] and _adj[_j, _k]:
                    _triangles.append((_i, _j, _k))
        _tets = []
        for (_i, _j, _k) in _triangles:
            for _l in range(_k + 1, N):
                if _adj[_i, _l] and _adj[_j, _l] and _adj[_k, _l]:
                    _tets.append((_i, _j, _k, _l))
        return {
            "N": N, "adj": _adj,
            "edges": _edges, "triangles": _triangles, "tets": _tets,
        }
    return (gen_complex,)


@app.cell(hide_code=True)
def _(np):
    def _find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(parent, rank, a, b):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    def perc_once(cx, p, dim, seed):
        """
        One bond-percolation trial on complex cx at probability p for k-simplices dim.
        Returns S = giant_component_vertices / N.
        """
        _N = cx["N"]
        _rng = np.random.default_rng(seed)
        _adj_p = np.zeros((_N, _N), dtype=bool)

        if dim == 1:
            for (_i, _j) in cx["edges"]:
                if _rng.random() < p:
                    _adj_p[_i, _j] = _adj_p[_j, _i] = True
        elif dim == 2:
            for (_i, _j, _k) in cx["triangles"]:
                if _rng.random() < p:
                    _adj_p[_i, _j] = _adj_p[_j, _i] = True
                    _adj_p[_i, _k] = _adj_p[_k, _i] = True
                    _adj_p[_j, _k] = _adj_p[_k, _j] = True
        else:
            for (_i, _j, _k, _l) in cx["tets"]:
                if _rng.random() < p:
                    for (_a, _b) in [(_i,_j),(_i,_k),(_i,_l),(_j,_k),(_j,_l),(_k,_l)]:
                        _adj_p[_a, _b] = _adj_p[_b, _a] = True

        # BFS component labeling
        _comp = np.zeros(_N, dtype=int)
        _c = 0
        for _start in range(_N):
            if _comp[_start] != 0:
                continue
            _c += 1
            _queue = [_start]
            _comp[_start] = _c
            while _queue:
                _v = _queue.pop(0)
                for _u in np.where(_adj_p[_v])[0]:
                    if _comp[_u] == 0:
                        _comp[_u] = _c
                        _queue.append(_u)

        if _c == 0:
            return 0.0
        _cnts = np.bincount(_comp[_comp > 0], minlength=_c + 1)[1:]
        return float(_cnts.max()) / _N if len(_cnts) > 0 else 0.0
    return (perc_once,)


@app.cell(hide_code=True)
def _(np, perc_once):
    def sweep(cx, p_arr, dim, n_rep):
        """
        Sweep over p_arr, returning S(p) and chi(p) averaged over n_rep realisations.
        chi = N * Var(S) / (mean(S) + 1e-12)
        """
        _N = cx["N"]
        _S = np.zeros(len(p_arr))
        _chi = np.zeros(len(p_arr))
        for _q, _p in enumerate(p_arr):
            _ss = np.array([perc_once(cx, _p, dim, _seed) for _seed in range(n_rep)])
            _ms = float(_ss.mean())
            _ms2 = float((_ss ** 2).mean())
            _var = max(0.0, _ms2 - _ms ** 2)
            _S[_q] = _ms
            _chi[_q] = _N * _var / (_ms + 1e-12)
        return _S, _chi
    return (sweep,)


@app.cell(hide_code=True)
def _(dim_k, gen_complex, n_rep, np, p_edge, sweep, N_vert):
    _cx   = gen_complex(N_vert, p_edge, seed=42)
    _p_arr = np.linspace(0.0, 1.0, 31)
    _S_arr, _chi_arr = sweep(_cx, _p_arr, dim_k, n_rep)

    _pc_idx  = int(np.argmax(_chi_arr))
    p_c      = float(_p_arr[_pc_idx])
    S_at_pc  = float(_S_arr[_pc_idx])
    chi_max  = float(_chi_arr.max())
    n_edges  = len(_cx["edges"])
    n_tri    = len(_cx["triangles"])
    n_tet    = len(_cx["tets"])
    S_full   = _S_arr.tolist()
    chi_full = _chi_arr.tolist()
    p_arr    = _p_arr.tolist()
    return (
        S_at_pc, S_full, chi_full, chi_max,
        n_edges, n_tet, n_tri, p_arr, p_c,
    )


@app.cell(hide_code=True)
def _(S_at_pc, chi_max, dim_k, mo, n_edges, n_rep, n_tet, n_tri, p_c, p_edge, N_vert):
    _dim_labels = {1: "edges (k=1)", 2: "triangles (k=2)", 3: "tetrahedra (k=3)"}
    mo.md(
        f"""
        | Parameter | Value |
        |-----------|-------|
        | Percolating | **{_dim_labels[dim_k]}** |
        | N vertices | {N_vert} |
        | p_edge | {p_edge:.2f} |
        | n_rep | {n_rep} |
        | p_c estimate | **{p_c:.2f}** |
        | S(p_c) | {S_at_pc:.3f} |
        | chi max | {chi_max:.1f} |
        | Complex | {n_edges} edges / {n_tri} triangles / {n_tet} tetrahedra |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Animated S(p) and chi(p) curves")
    return


@app.cell(hide_code=True)
def _(S_at_pc, S_full, canvas_anim, chi_full, mo, p_arr, p_c):

    _S_max  = max(S_full) if S_full else 1.0
    _chi_max_v = max(chi_full) if chi_full else 1.0

    _spec = {
        "frames": len(p_arr),
        "dt": 120,
        "p_arr": p_arr,
        "panels": [
            {
                "kind": "sweep_curve",
                "data": S_full,
                "ylabel": "S(p)", "xlabel": "bond probability p",
                "title": "Giant component fraction S(p)",
                "color": "#a594f2",
                "pc": p_c,
            },
            {
                "kind": "sweep_curve",
                "data": chi_full,
                "ylabel": "χ(p)", "xlabel": "bond probability p",
                "title": "Susceptibility chi(p)",
                "color": "#e6a356",
                "pc": p_c,
            },
        ],
        "layout": [
            [0.0,  0.0, 0.50, 1.0],
            [0.50, 0.0, 0.50, 1.0],
        ],
    }

    mo.vstack([
        mo.md(
            f"**p_c** = {p_c:.2f} &nbsp;|&nbsp; S(p_c) = {S_at_pc:.3f} &nbsp;|&nbsp; "
            "Left: S(p) fills in point by point as p increases. "
            "Right: chi(p) peaks at the critical threshold."
        ),
        canvas_anim(_spec, height="460px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Graph animation: growing giant component")
    return


@app.cell(hide_code=True)
def _(np):
    def build_geo_complex(N=60, radius=0.18, seed=1):
        """
        Geometric random graph: vertices at random positions in [0,1]^2,
        edges between pairs closer than radius.
        Returns positions, edges, triangles.
        """
        _rng = np.random.default_rng(seed)
        _pos = _rng.random((N, 2))
        _edges = []
        for _i in range(N):
            for _j in range(_i + 1, N):
                if np.hypot(*(_pos[_i] - _pos[_j])) < radius:
                    _edges.append((_i, _j))
        _nbr = {_i: set() for _i in range(N)}
        for _i, _j in _edges:
            _nbr[_i].add(_j); _nbr[_j].add(_i)
        _triangles = []
        for _i, _j in _edges:
            for _k in (_nbr[_i] & _nbr[_j]):
                if _k > _j:
                    _triangles.append((_i, _j, _k))
        return _pos, _edges, _triangles

    def giant_mask_uf(N, active_edge_indices, edges):
        """Union-find giant component mask."""
        _parent = list(range(N))
        _rank = [0] * N

        def _find(x):
            while _parent[x] != x:
                _parent[x] = _parent[_parent[x]]
                x = _parent[x]
            return x

        for _ei in active_edge_indices:
            _a, _b = edges[_ei]
            _ra, _rb = _find(_a), _find(_b)
            if _ra != _rb:
                if _rank[_ra] < _rank[_rb]:
                    _ra, _rb = _rb, _ra
                _parent[_rb] = _ra
                if _rank[_ra] == _rank[_rb]:
                    _rank[_ra] += 1

        _roots = np.array([_find(_i) for _i in range(N)])
        _cnts = np.bincount(_roots, minlength=N)
        _big = int(_cnts.argmax())
        _mask = _roots == _big
        return _mask, float(_mask.sum()) / N
    return build_geo_complex, giant_mask_uf


@app.cell(hide_code=True)
def _(build_geo_complex, canvas_anim, giant_mask_uf, mo, np):
    _GN = 60
    _GR = 0.19
    _pos_g, _edges_g, _tris_g = build_geo_complex(N=_GN, radius=_GR, seed=3)
    _E = len(_edges_g)
    _ei_map = {tuple(sorted(_e)): _idx for _idx, _e in enumerate(_edges_g)}

    _rng_g = np.random.default_rng(11)
    _ue_g  = _rng_g.random(_E)
    _ut_g  = _rng_g.random(len(_tris_g))

    def _active_edges_at_p(p, use_dim):
        if use_dim == 1:
            return list(np.where(_ue_g < p)[0])
        _on = set()
        for _ti, (_i, _j, _k) in enumerate(_tris_g):
            if _ut_g[_ti] < p:
                for _pair in ((_i,_j), (_i,_k), (_j,_k)):
                    _on.add(_ei_map[tuple(sorted(_pair))])
        return sorted(_on)

    _pv_g = np.linspace(0.0, 1.0, 50)
    _fa_g, _fg_g = [], []
    for _pg in _pv_g:
        _ae = _active_edges_at_p(_pg, 1)
        _gmask, _ = giant_mask_uf(_GN, _ae, _edges_g)
        _a_arr = np.zeros(_E, dtype=int); _a_arr[_ae] = 1
        _fa_g.append(_a_arr.tolist())
        _fg_g.append(_gmask.astype(int).tolist())

    _spec_g = {
        "frames": len(_pv_g), "dt": 85,
        "panels": [{"kind": "geo_graph"}],
        "layout": [[0.0, 0.0, 1.0, 1.0]],
        "pos": np.round(_pos_g, 3).tolist(),
        "edges": [list(_e) for _e in _edges_g],
        "active": _fa_g, "giant": _fg_g,
        "pvals": np.round(_pv_g, 3).tolist(),
    }

    mo.vstack([
        mo.md(
            "**Edge percolation, live.** Edges switch on as p rises. "
            "The giant component (largest cluster) is shown in violet. "
            "Near p_c it abruptly spans the network."
        ),
        canvas_anim(_spec_g, height="460px"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How it works

        **Complex construction.** Start with N vertices. Add each of the $\binom{N}{2}$ possible
        edges independently with probability $p_\text{edge}$, producing an Erdos-Renyi random
        graph $G(N, p_\text{edge})$. Every triangle (3-clique) in the graph becomes a 2-simplex,
        and every $K_4$ subgraph becomes a 3-simplex. No extra randomness beyond the initial
        edge coin flips is needed.

        **Bond percolation on $k$-simplices.** Each simplex of dimension $k$ is independently
        occupied with probability $p$:

        - $k=1$: each edge is occupied with probability $p$ (standard bond percolation).
        - $k=2$: each triangle is occupied with probability $p$; when it fires, all three of its
          edges become active simultaneously.
        - $k=3$: each tetrahedron fires with probability $p$, activating all six of its edges.

        This cooperative structure means that for $k \geq 2$ a single coin flip brings three or
        six bonds at once, raising the effective connectivity per trial and shifting $p_c$.

        **Observables.** Connected components are found by BFS on the activated edge graph. The
        order parameter is $S(p) = $ (vertices in largest component) $/ N$, averaged over
        $n_\text{rep}$ independent realisations. The susceptibility is
        $\chi(p) = N \operatorname{Var}(S) / (\langle S \rangle + \epsilon)$, which peaks sharply
        at the critical threshold. The location of the $\chi$ peak gives the finite-size estimate
        of $p_c$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
