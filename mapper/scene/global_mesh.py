"""The persistent welded map.

Patches used to be kept as a list and offset-concatenated into a throwaway
`TriangleModel` every keyframe, so a seam observed twice existed twice: two sets
of vertices at the same place, two boundaries that never closed. `seed_patch_cdt`
has always returned the table needed to fix that -- `ids_full[k] >= 0` means new
vertex k *is* global vertex `ids_full[k]`, already positioned bit-identically --
and this is what consumes it.

Vertex numbering is owned here and is append-only for the life of a session. That
is deliberate: `ids_full` is captured against the mesh at seed time and applied at
weld time, so any renumbering in between would silently land welds on the wrong
vertices. Pruning must therefore mark vertices dead rather than renumber, and
`weld` asserts the version it was seeded against.
"""
import math
from collections import defaultdict

import numpy as np
import torch

from scene.patch import boundary_loops, edge_key, halfedge_maps
from utils.timing import count, tic

# packing limit for the duplicate-face key; asserted on use
_VBITS = 21
_VMAX = 1 << _VBITS


class PatchRecord:
    """Bookkeeping for one welded patch, for per-patch scheduling."""

    def __init__(self, pid, kf_index, age):
        self.pid = pid
        self.kf_index = kf_index
        self.age = age
        self.n_faces = 0
        self.n_vertices = 0          # vertices this patch created, not shares

    def __repr__(self):
        return (f"PatchRecord(pid={self.pid}, kf={self.kf_index}, "
                f"faces={self.n_faces}, new_verts={self.n_vertices})")


class NonManifold(RuntimeError):
    pass


class GlobalMesh:
    def __init__(self, device="cuda"):
        self.device = device
        self.vertices = torch.zeros((0, 3), dtype=torch.float32, device=device)
        self.colors = torch.zeros((0, 3), dtype=torch.float32, device=device)
        self.faces = np.zeros((0, 3), np.int64)
        self.face_patch = np.zeros(0, np.int32)
        self.vertex_patch = np.zeros(0, np.int32)
        self.patches = []
        self.version = 0

        self._edge_faces = {}                 # packed edge key -> [face index]
        self._n_boundary = 0                  # edges with exactly one face
        self._face_keys = set()               # packed sorted triple, dedupe guard
        self._loops_cache = None
        self._loops_version = -1
        # running totals; a skipped split is a T-junction we chose not to make
        self.n_splits_applied = 0
        self.n_splits_skipped = 0
        self.n_faces_rejected = 0

    # -- basic queries -----------------------------------------------------

    def __len__(self):
        return len(self.faces)

    @property
    def n_vertices(self):
        return len(self.vertices)

    def is_empty(self):
        return len(self.faces) == 0

    def boundary_edge_count(self):
        return self._n_boundary

    def boundary_loops(self):
        """Oriented boundary rings, as global vertex indices.

        Recomputed from the face array and cached on `version`. The incremental
        edge counts below are exact and O(new faces), but *orienting* the rings
        needs the face fan around each boundary vertex to resolve pinch
        vertices, which welding creates routinely -- so this stays a recompute
        rather than a subtly-wrong shortcut. Cost is O(F log F); it is measured
        under the `boundary_rings` timer.
        """
        if self._loops_version == self.version and self._loops_cache is not None:
            return self._loops_cache
        with tic("boundary_rings"):
            loops = boundary_loops(self.faces, len(self.vertices))
        self._loops_cache = loops
        self._loops_version = self.version
        count("n_boundary_rings", len(loops))
        return loops

    # -- per-patch scheduling ---------------------------------------------

    def faces_of(self, pid):
        """Indices of the faces contributed by patch `pid`."""
        pids = np.atleast_1d(pid)
        return np.flatnonzero(np.isin(self.face_patch, pids))

    def vertex_mask(self, pid):
        """(V,) bool: vertices touched by patch `pid`'s faces.

        Seam vertices are shared, so masks for adjacent patches overlap. That is
        the truth of a welded mesh -- `face_patch` is the authoritative record,
        this is a convenience for per-vertex parameters like opacity.
        """
        m = torch.zeros(len(self.vertices), dtype=torch.bool, device=self.device)
        f = self.faces_of(pid)
        if len(f):
            idx = torch.from_numpy(np.unique(self.faces[f])).to(self.device)
            m[idx] = True
        return m

    def active_patches(self, kf_index, window):
        """Patches seeded within `window` keyframes of `kf_index`."""
        return np.array([p.pid for p in self.patches
                         if kf_index - p.kf_index <= window], np.int64)

    def patch_ages(self):
        return np.array([p.age for p in self.patches], np.int64)

    # -- edge bookkeeping --------------------------------------------------

    @staticmethod
    def _face_edge_keys(F):
        return edge_key(F[:, [0, 1, 2]], F[:, [1, 2, 0]])

    @staticmethod
    def _face_key(F):
        s = np.sort(F, axis=1)
        assert s.max(initial=0) < _VMAX, "vertex count exceeds the face-key packing"
        return (s[:, 0] << (2 * _VBITS)) | (s[:, 1] << _VBITS) | s[:, 2]

    def _edge_count_of(self, key):
        return len(self._edge_faces.get(key, ()))

    def _attach(self, fi, f):
        """Record face index `fi` (with vertices `f`) against its three edges."""
        for key in self._face_edge_keys(np.asarray(f)[None, :])[0].tolist():
            lst = self._edge_faces.setdefault(key, [])
            was = len(lst)
            lst.append(fi)
            self._n_boundary += (len(lst) == 1) - (was == 1)

    def _detach(self, fi, f):
        for key in self._face_edge_keys(np.asarray(f)[None, :])[0].tolist():
            lst = self._edge_faces.get(key)
            if not lst or fi not in lst:
                continue
            was = len(lst)
            lst.remove(fi)
            self._n_boundary += (len(lst) == 1) - (was == 1)
            if not lst:
                self._edge_faces.pop(key, None)

    def _apply_splits(self, splits):
        """Split map edges that a new seam lands inside. -> {patch vertex: gid}.

        Each target is a *boundary* edge, so exactly one face uses it and the
        surgery is local: rebuild that face as a fan over the split chain. The
        children keep the parent's patch id -- the surface is still the older
        patch's, it just has more vertices on its rim now.
        """
        out = {}
        if not splits:
            return out
        by_edge = defaultdict(list)
        for (k, a, b, t) in splits:
            # measure t from the lower vertex id so multiple splits sort together
            by_edge[(a, b) if a <= b else (b, a)].append(
                (t if a <= b else 1.0 - t, k))

        n_done = n_skip = 0
        for (a, b), items in by_edge.items():
            key = int(edge_key(a, b))
            faces = self._edge_faces.get(key, [])
            if len(faces) != 1:
                # not on the frontier any more -- an earlier split in this same
                # batch may already have replaced it. Degrade to a T-junction
                # rather than corrupt the face array.
                n_skip += len(items)
                continue
            fi = faces[0]
            f = self.faces[fi].copy()
            items.sort()

            va = self.vertices[a]
            vb = self.vertices[b]
            ca = self.colors[a]
            cb = self.colors[b]
            gids = []
            for t, k in items:
                gid = len(self.vertices)
                self.vertices = torch.cat([self.vertices, (va * (1 - t) + vb * t)[None]])
                self.colors = torch.cat([self.colors, (ca * (1 - t) + cb * t)[None]])
                self.vertex_patch = np.concatenate(
                    [self.vertex_patch, np.int32([self.face_patch[fi]])])
                gids.append(gid)
                out[k] = gid

            # locate the edge inside the face and keep its winding
            i = next(j for j in range(3)
                     if {int(f[j]), int(f[(j + 1) % 3])} == {int(a), int(b)})
            u, w, c = int(f[i]), int(f[(i + 1) % 3]), int(f[(i + 2) % 3])
            chain = [u] + (gids if u == a else gids[::-1]) + [w]
            new = np.array([[chain[j], chain[j + 1], c]
                            for j in range(len(chain) - 1)], np.int64)

            self._detach(fi, f)
            self._face_keys.discard(int(self._face_key(f[None, :])[0]))
            pid = self.face_patch[fi]
            self.faces[fi] = new[0]                       # reuse the slot
            self._attach(fi, new[0])
            self._face_keys.add(int(self._face_key(new[0][None, :])[0]))
            if len(new) > 1:
                base = len(self.faces)
                self.faces = np.concatenate([self.faces, new[1:]])
                self.face_patch = np.concatenate(
                    [self.face_patch, np.full(len(new) - 1, pid, np.int32)])
                self._face_keys.update(self._face_key(new[1:]).tolist())
                for j, nf in enumerate(new[1:]):
                    self._attach(base + j, nf)
            n_done += len(items)

        count("n_edge_splits_applied", n_done)
        count("n_edge_splits_skipped", n_skip)
        self.n_splits_applied += n_done
        self.n_splits_skipped += n_skip
        return out

    def _reject_bad_faces(self, F, pid):
        """Drop degenerate, duplicate, and non-manifold-inducing faces.

        A chord of the seam can join two vertices whose edge is already interior
        to the map; welding a face onto it would give that edge three incident
        faces. `halfedge_maps` treats such an edge as boundary, so the damage
        would be invisible until the next keyframe's boolean subtracted the
        wrong region. Cheaper to refuse the face and count it.
        """
        if len(F) == 0:
            return F
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        n_degen = int((~keep).sum())
        F = F[keep]

        n_dup = 0
        if len(F):
            fk = self._face_key(F)
            fresh = np.zeros(len(F), bool)
            fresh[np.unique(fk, return_index=True)[1]] = True     # within batch
            if self._face_keys:
                seen = np.fromiter((k in self._face_keys for k in fk.tolist()),
                                   bool, len(fk))
                fresh &= ~seen                                    # against the map
            n_dup = int((~fresh).sum())
            F = F[fresh]

        n_nm = 0
        if len(F):
            keys = self._face_edge_keys(F)
            existing = np.array([[self._edge_count_of(int(k)) for k in row]
                                 for row in keys], np.int64)
            uk, inv, cnt = np.unique(keys.ravel(), return_inverse=True,
                                     return_counts=True)
            within = cnt[inv].reshape(keys.shape)
            bad = ((existing + within) > 2).any(1)
            n_nm = int(bad.sum())
            F = F[~bad]

        if n_degen or n_dup or n_nm:
            self.n_faces_rejected += n_degen + n_dup + n_nm
            print(f"[weld] patch {pid}: dropped {n_degen} degenerate, "
                  f"{n_dup} duplicate, {n_nm} non-manifold faces")
        count("n_faces_degenerate", n_degen)
        count("n_faces_duplicate", n_dup)
        count("n_faces_nonmanifold", n_nm)
        return F

    # -- the weld ----------------------------------------------------------

    def weld(self, patch, ids, kf_index, seed_version=None, splits=None):
        """Integrate `patch` into the map. Returns its PatchRecord.

        `ids[k] >= 0` names an existing global vertex; those are reused rather
        than duplicated, which is what actually closes the seam. Everything else
        is appended.
        """
        if seed_version is not None and seed_version != self.version:
            raise RuntimeError(
                f"patch was seeded against mesh version {seed_version} but the "
                f"mesh is now at {self.version}; the ids may name different "
                f"vertices than they did at seed time")

        with tic("weld"):
            pid = len(self.patches)
            rec = PatchRecord(pid, kf_index, getattr(patch, "age", kf_index))

            n = len(patch.vertices)
            ids = np.asarray(ids, np.int64)
            assert len(ids) == n, f"ids has {len(ids)} entries for {n} vertices"
            if len(ids) and ids.max() >= len(self.vertices):
                raise RuntimeError(
                    f"weld id {ids.max()} is out of range for a mesh with "
                    f"{len(self.vertices)} vertices")

            # split the map's own edges first: that both creates the vertices
            # the new seam needs and removes the T-junctions it would otherwise
            # leave behind
            split_gids = self._apply_splits(splits)

            remap = np.full(n, -1, np.int64)
            have = ids >= 0
            remap[have] = ids[have]
            for k, gid in split_gids.items():
                remap[k] = gid
                have[k] = True

            new = remap < 0
            n_new = int(new.sum())
            base = len(self.vertices)
            remap[new] = base + np.arange(n_new)

            if n_new:
                sel = torch.from_numpy(np.flatnonzero(new)).to(self.device)
                self.vertices = torch.cat(
                    [self.vertices, patch.vertices.detach()[sel].to(self.device)])
                self.colors = torch.cat(
                    [self.colors, patch.colors.detach()[sel].to(self.device)])
                # creator wins: a welded vertex keeps the patch that made it, so
                # the age render and per-patch masks stay stable as neighbours
                # are added around it
                self.vertex_patch = np.concatenate(
                    [self.vertex_patch, np.full(n_new, pid, np.int32)])
            rec.n_vertices = n_new

            F = remap[patch.faces.detach().cpu().numpy().astype(np.int64)]
            F = self._reject_bad_faces(F, pid)
            if len(F):
                base_f = len(self.faces)
                self._face_keys.update(self._face_key(F).tolist())
                self.faces = np.concatenate([self.faces, F])
                self.face_patch = np.concatenate(
                    [self.face_patch, np.full(len(F), pid, np.int32)])
                for i, f in enumerate(F):
                    self._attach(base_f + i, f)
            rec.n_faces = len(F)

            self.patches.append(rec)
            self.version += 1

        count("n_weld_reused", int(have.sum()))
        count("n_weld_new", n_new)
        count("mesh_vertices", len(self.vertices))
        count("mesh_faces", len(self.faces))
        count("mesh_boundary_edges", self.boundary_edge_count())
        return rec

    # -- handing the map to the renderer ------------------------------------

    def to_triangle_model(self, colors_mode="rgb", sigma=0.001, opacity=0.99,
                          age_to_color=None):
        """Build a TriangleModel over the current mesh.

        `patch_id` goes in as a per-triangle buffer so that anything scheduled
        per patch survives into the model. In "age" mode a vertex is coloured by
        the age of the patch that *created* it, so a welded seam keeps the older
        patch's colour instead of flickering to the newest one.
        """
        from scene.triangle_model import TriangleModel      # cuda import; late

        colors = self.colors
        if colors_mode == "age" and age_to_color is not None and len(self.patches):
            # vertex_patch already IS the patch ordinal, and a welded vertex keeps
            # its creator, so the age render shows when each bit of surface first
            # appeared rather than when it was last touched
            per_vertex = self.vertex_patch.astype(np.int64)
            uniq = np.unique(per_vertex)                       # one colour per age
            lut = torch.stack([age_to_color(int(a)) for a in uniq]).to(self.device)
            colors = lut[torch.from_numpy(
                np.searchsorted(uniq, per_vertex)).to(self.device)]

        with tic("to_triangle_model"):
            tm = TriangleModel(sh_degree=0)
            tm.populate_triangle_model(
                self.vertices,
                torch.from_numpy(self.faces).to(self.device),
                colors, sigma,
                patch_id=torch.from_numpy(self.face_patch).to(self.device))
            with torch.no_grad():
                # vertex_weight is a logit; the caller gives a real opacity
                o = float(min(max(opacity, 1e-4), 1 - 1e-6))
                tm.vertex_weight.fill_(math.log(o / (1.0 - o)))
        return tm

    def append_into_model(self, tm, v_before, opacity=0.99):
        """Extend a live TriangleModel with everything welded since `v_before`.

        The alternative -- rebuilding the model from the mesh -- throws away the
        optimiser with it, so momentum never accumulates and the learning-rate
        schedule restarts. `cat_tensors_to_optimizer` grows each parameter and
        pads Adam's exp_avg/exp_avg_sq with zeros for the new rows, so existing
        geometry keeps its state and new geometry starts clean.

        Faces are not parameters, so the topology is copied over wholesale --
        edge splits rewrite an existing face in place, which an append could not
        express.
        """
        from utils.general_utils import inverse_sigmoid
        from utils.sh_utils import RGB2SH

        dev = tm.vertices.device if len(tm.vertices) else self.device
        n_new = len(self.vertices) - v_before
        if n_new > 0:
            new_v = self.vertices[v_before:].detach().clone().to(dev)
            new_c = RGB2SH(self.colors[v_before:].detach().clone().to(dev))
            o = float(min(max(opacity, 1e-4), 1 - 1e-6))
            d = {
                "vertices": new_v,
                "vertex_weight": torch.full((n_new, 1), math.log(o / (1 - o)),
                                            device=dev),
                "f_dc": new_c[:, None, :],                       # (n,1,3)
                "f_rest": torch.zeros((n_new, tm._features_rest.shape[1], 3),
                                      device=dev),
            }
            got = tm.cat_tensors_to_optimizer(d)
            tm.vertices = got["vertices"]
            tm.vertex_weight = got["vertex_weight"]
            tm._features_dc = got["f_dc"]
            tm._features_rest = got["f_rest"]

        f_before = tm._triangle_indices.shape[0]
        tm._triangle_indices = torch.from_numpy(self.faces).to(torch.int32).to(dev)
        tm.face_patch = torch.from_numpy(self.face_patch).to(torch.int32).to(dev)
        n_faces = tm._triangle_indices.shape[0]
        for name in ("image_size", "importance_score"):
            old = getattr(tm, name, None)
            if not torch.is_tensor(old) or old.numel() == 0:
                setattr(tm, name, torch.zeros(n_faces, dtype=torch.float, device=dev))
            elif old.shape[0] < n_faces:                 # keep the running maxima
                setattr(tm, name, torch.cat(
                    [old, torch.zeros(n_faces - old.shape[0], dtype=torch.float,
                                      device=dev)]))
            elif old.shape[0] > n_faces:
                setattr(tm, name, old[:n_faces])
        count("n_model_verts_appended", max(n_new, 0))
        count("n_model_faces_appended", n_faces - f_before)
        return tm

    def sync_from(self, tm):
        """Pull optimised vertex positions and colours back out of a model.

        Optimisation runs on the TriangleModel, so without this the next
        `to_triangle_model` would rebuild from stale geometry and throw the
        result away. Face topology is owned here and is never taken back from
        the model -- if the model has pruned or densified, the counts differ and
        this refuses rather than mismatching vertices to faces.
        """
        from utils.sh_utils import SH2RGB

        if len(tm.vertices) != len(self.vertices):
            raise RuntimeError(
                f"model has {len(tm.vertices)} vertices, mesh has "
                f"{len(self.vertices)}; the model was pruned or densified and "
                f"the mesh cannot absorb that yet")
        with torch.no_grad():
            self.vertices = tm.vertices.detach().clone()
            self.colors = SH2RGB(
                tm._features_dc.detach()[:, 0, :]).clamp(0, 1).clone()
        self.version += 1
        return self

    # -- verification ------------------------------------------------------

    def verify(self, full=False, pos_tol=1e-6):
        """Invariants. `full` also recomputes the edge counts from scratch."""
        problems = []
        if len(self.face_patch) != len(self.faces):
            problems.append("face_patch length != face count")
        if len(self.vertex_patch) != len(self.vertices):
            problems.append("vertex_patch length != vertex count")
        if len(self.faces) and self.faces.max() >= len(self.vertices):
            problems.append("a face references a vertex that does not exist")
        if len(self.face_patch) and self.face_patch.max() >= len(self.patches):
            problems.append("face_patch names a patch that does not exist")

        if full and len(self.faces):
            fresh = {}
            for key, c in zip(*np.unique(self._face_edge_keys(self.faces).ravel(),
                                         return_counts=True)):
                fresh[int(key)] = int(c)
            live = {k: len(v) for k, v in self._edge_faces.items() if v}
            if fresh != live:
                problems.append("incremental edge counts disagree with a recompute")
            if self._n_boundary != sum(1 for v in live.values() if v == 1):
                problems.append("boundary edge counter drifted")

            # the crack detector: two distinct vertices at one position means a
            # seam was duplicated instead of welded
            v = self.vertices.detach().cpu().numpy()
            q = np.round(v / pos_tol).astype(np.int64)
            uniq = len(np.unique(q, axis=0))
            if uniq != len(v):
                problems.append(
                    f"{len(v) - uniq} coincident vertex positions -- seams are "
                    f"being duplicated, not welded")
        return problems

    def stats(self):
        return (f"mesh: {len(self.vertices)} verts, {len(self.faces)} faces, "
                f"{len(self.patches)} patches, "
                f"{self.boundary_edge_count()} boundary edges")
