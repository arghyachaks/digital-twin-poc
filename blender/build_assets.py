"""
Refinery Digital Twin - Blender asset builder
=============================================
Run headless:
    blender.exe -b --factory-startup -P blender/build_assets.py

Produces (relative to the project root):
    Content_Source/Textures/T_<Set>_{BaseColor,Normal,ORM}.png   (tileable, <= 2048x2048)
    Content_Source/Meshes/SM_<Asset>.fbx                        (one hero asset per file, origin at base)
    Content_Source/Blender/RefineryAssets.blend                 (all assets laid out for editing)

Units: metres in Blender (exported so Unreal receives cm at correct size).
Material slot names match keys in config/materials.json so Unreal can bind them.
ORM packing: R = ambient occlusion, G = roughness, B = metallic. Normal maps are DirectX (green down) for Unreal.
"""
import bpy
import bmesh
import json
import math
import os
import sys
import time

import numpy as np
from mathutils import Matrix, Vector

# --------------------------------------------------------------------------- paths / config
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CFG = json.load(open(os.path.join(ROOT, "config", "materials.json")))
TEX_RES = min(int(CFG.get("max_texture_size", 2048)), 2048)  # hard cap 2K
OUT_TEX = os.path.join(ROOT, "Content_Source", "Textures")
OUT_MESH = os.path.join(ROOT, "Content_Source", "Meshes")
OUT_BLEND = os.path.join(ROOT, "Content_Source", "Blender")
OUT_WEB = os.path.join(ROOT, "Content_Source", "Web")
for d in (OUT_TEX, OUT_MESH, OUT_BLEND, OUT_WEB):
    os.makedirs(d, exist_ok=True)

UV_TILE_M = 2.0  # one texture repeat every 2 m (box projection); per-material scale tweaks in Unreal
TAU = math.tau


def log(msg):
    print(f"[TwinBuilder] {msg}", flush=True)


# =========================================================================== TEXTURES
# Tileable fractal noise generated with numpy inside Blender, so every map wraps seamlessly.

def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def periodic_noise(res, freq, rng):
    """Value noise on a freq x freq lattice that wraps exactly at the image edge."""
    grid = rng.random((freq, freq)).astype(np.float32)
    coords = np.arange(res, dtype=np.float32) * freq / res
    i0 = np.floor(coords).astype(np.int32)
    t = _smooth(coords - i0)
    i0 %= freq
    i1 = (i0 + 1) % freq
    # interpolate rows then columns
    a = grid[i0][:, i0] * (1 - t)[None, :] + grid[i0][:, i1] * t[None, :]
    b = grid[i1][:, i0] * (1 - t)[None, :] + grid[i1][:, i1] * t[None, :]
    return a * (1 - t)[:, None] + b * t[:, None]


def fbm(res, base_freq, octaves, rng, gain=0.5):
    out = np.zeros((res, res), np.float32)
    amp, total, f = 1.0, 0.0, base_freq
    for _ in range(octaves):
        if f > res:
            break
        out += periodic_noise(res, f, rng) * amp
        total += amp
        amp *= gain
        f *= 2
    return out / total


def normal_from_height(h, strength):
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * strength
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * strength
    n = np.stack([-dx, dy, np.ones_like(h)], axis=-1)  # dy sign -> DirectX convention after flip below
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n[..., 1] *= -1.0  # DirectX green
    return n * 0.5 + 0.5


def save_png(name, rgb, srgb=True):
    """rgb: (res,res,3) float 0..1. Saved as 8-bit PNG via Blender image API."""
    res = rgb.shape[0]
    assert res <= 2048, "texture cap is 2048"
    rgba = np.concatenate([np.clip(rgb, 0, 1), np.ones((res, res, 1), np.float32)], axis=-1)
    img = bpy.data.images.new(name, width=res, height=res, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    path = os.path.join(OUT_TEX, name + ".png")
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return path


def gray(x):
    return np.repeat(x[..., None], 3, axis=-1)


def orm(ao, rough, metal):
    return np.stack([ao, rough, metal], axis=-1)


def make_texture_sets(res):
    rng = np.random.default_rng(42)
    sets = {}

    # ---- PaintedSteel: neutral light paint (tinted per instance in UE), chips exposing dark metal, grime
    grime = fbm(res, 4, 7, rng)
    chips_n = fbm(res, 16, 5, rng)
    chips = np.clip((chips_n - 0.68) * 12, 0, 1)
    paint = 0.82 + (grime - 0.5) * 0.12
    base = gray(paint) * (1 - chips[..., None]) + gray(np.full_like(paint, 0.25)) * chips[..., None]
    base *= (0.85 + 0.15 * grime)[..., None]
    rough = np.clip(0.42 + (grime - 0.5) * 0.25 + chips * 0.2, 0, 1)
    metal = chips * 0.9
    height = fbm(res, 64, 4, rng) * 0.3 - chips * 0.6
    sets["PaintedSteel"] = (base, normal_from_height(height, 6.0), orm(np.ones_like(rough), rough, metal))

    # ---- Galvanized: zinc spangle mottling, fully metallic
    sp = fbm(res, 32, 6, rng)
    sp2 = fbm(res, 8, 3, rng)
    base = gray(0.62 + (sp - 0.5) * 0.25 + (sp2 - 0.5) * 0.15)
    rough = np.clip(0.38 + (sp - 0.5) * 0.35, 0.1, 1)
    sets["Galvanized"] = (base, normal_from_height(sp * 0.2, 4.0), orm(np.ones_like(rough), rough, np.ones_like(rough)))

    # ---- Concrete: aggregate speckle, stains, expansion joints at tile edges
    c1 = fbm(res, 8, 8, rng)
    speck = np.clip((fbm(res, 128, 2, rng) - 0.6) * 4, 0, 1)
    stains = np.clip((fbm(res, 2, 4, rng) - 0.45) * 2, 0, 1)
    v = np.arange(res) / res
    joint1d = np.minimum(v, 1 - v) * res
    joint = np.clip(1 - np.minimum(joint1d[None, :], joint1d[:, None]) / 6.0, 0, 1)
    tone = 0.55 + (c1 - 0.5) * 0.2 - speck * 0.08 - stains * 0.12 - joint * 0.35
    base = np.stack([tone * 1.0, tone * 0.98, tone * 0.94], axis=-1)
    rough = np.clip(0.85 + (c1 - 0.5) * 0.1 - stains * 0.15, 0, 1)
    ao = np.clip(1 - joint * 0.7, 0, 1)
    sets["Concrete"] = (base, normal_from_height(c1 * 0.4 + speck * 0.2 - joint, 5.0), orm(ao, rough, np.zeros_like(rough)))

    # ---- Cladding: aluminium insulation jacket, overlap seams every 1/4 of the tile vertically
    seam_pos = (np.arange(res) / res * 4.0) % 1.0
    seam = np.clip(1 - np.abs(seam_pos - 0.5) * res / 4 / 3.0, 0, 1)  # thin ridge
    seam2d = np.repeat(seam[:, None], res, axis=1)
    wobble = fbm(res, 8, 5, rng)
    base = gray(0.72 + (wobble - 0.5) * 0.12 - seam2d * 0.15)
    rough = np.clip(0.28 + (wobble - 0.5) * 0.2 + seam2d * 0.1, 0, 1)
    sets["Cladding"] = (base, normal_from_height(seam2d * 1.5 + wobble * 0.15, 3.0), orm(1 - seam2d * 0.3, rough, np.ones_like(rough)))

    # ---- Grating: bar grating, 16 bearing bars x 4 cross bars per tile, holes darkened (AO)
    u = (np.arange(res) / res * 16.0) % 1.0
    w = (np.arange(res) / res * 4.0) % 1.0
    bars = (np.abs(u - 0.5) > 0.38).astype(np.float32)
    cross = (np.abs(w - 0.5) > 0.47).astype(np.float32)
    solid = np.clip(bars[None, :] + cross[:, None], 0, 1)
    n = fbm(res, 16, 4, rng)
    base = gray(0.08 + solid * (0.75 + (n - 0.5) * 0.2))
    rough = 0.5 + (n - 0.5) * 0.2 + (1 - solid) * 0.4
    sets["Grating"] = (base, normal_from_height(solid * 2.0, 3.0), orm(0.15 + solid * 0.85, np.clip(rough, 0, 1), solid * 0.7))

    # ---- Refractory / heat-stained furnace casing: dark steel with heat tint and soot
    h1 = fbm(res, 4, 7, rng)
    soot = np.clip((fbm(res, 3, 5, rng) - 0.4) * 2, 0, 1)
    base = np.stack([0.20 + h1 * 0.10, 0.17 + h1 * 0.06, 0.15 + h1 * 0.04], axis=-1) * (1 - soot[..., None] * 0.6)
    rough = np.clip(0.6 + soot * 0.3, 0, 1)
    sets["Refractory"] = (base, normal_from_height(h1 * 0.5, 4.0), orm(np.ones_like(rough), rough, np.full_like(rough, 0.5)))

    for name, (bc, nrm, ormap) in sets.items():
        save_png(f"T_{name}_BaseColor", bc, srgb=True)
        save_png(f"T_{name}_Normal", nrm, srgb=False)
        save_png(f"T_{name}_ORM", ormap, srgb=False)
        log(f"texture set {name} @ {res}px")


# =========================================================================== MESH BUILDER

AXIS_ROT = {
    "Z": Matrix.Identity(4),
    "X": Matrix.Rotation(math.radians(90), 4, "Y"),
    "Y": Matrix.Rotation(math.radians(-90), 4, "X"),
}


class Asset:
    """Accumulates primitives into one bmesh with per-face material indices."""

    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.mats = []  # material keys in slot order

    def _mi(self, mat):
        if mat not in CFG["materials"]:
            raise KeyError(f"unknown material {mat}")
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _tag(self, verts, mat):
        idx = self._mi(mat)
        faces = {f for v in verts for f in v.link_faces}
        for f in faces:
            f.material_index = idx

    # ---- primitives -----------------------------------------------------
    def cyl(self, r, h, loc, mat, axis="Z", seg=32, r2=None, caps=True):
        m = Matrix.Translation(Vector(loc)) @ AXIS_ROT[axis]
        res = bmesh.ops.create_cone(self.bm, cap_ends=caps, cap_tris=False, segments=seg,
                                    radius1=r, radius2=r if r2 is None else r2, depth=h, matrix=m)
        self._tag(res["verts"], mat)

    def box(self, size, loc, mat, rot_z=0.0):
        m = Matrix.Translation(Vector(loc)) @ Matrix.Rotation(rot_z, 4, "Z") @ Matrix.Diagonal(Vector((*size, 1.0)))
        res = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        self._tag(res["verts"], mat)

    def sphere(self, r, loc, mat, scale=(1, 1, 1), seg=32, rings=16):
        m = Matrix.Translation(Vector(loc)) @ Matrix.Diagonal(Vector((*scale, 1.0)))
        res = bmesh.ops.create_uvsphere(self.bm, u_segments=seg, v_segments=rings, radius=r, matrix=m)
        self._tag(res["verts"], mat)

    def torus(self, R, r, loc, mat, axis="Z", seg_R=48, seg_r=10, arc=TAU, start=0.0):
        rot = AXIS_ROT[axis]
        full = abs(arc - TAU) < 1e-6
        nR = seg_R if full else seg_R + 1
        rings = []
        for i in range(nR):
            a = start + arc * i / seg_R
            ring = []
            for j in range(seg_r):
                b = TAU * j / seg_r
                p = Vector(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b)))
                ring.append(self.bm.verts.new((Matrix.Translation(Vector(loc)) @ rot) @ p))
            rings.append(ring)
        new = []
        for i in range(nR if full else nR - 1):
            a, b = rings[i], rings[(i + 1) % nR]
            for j in range(seg_r):
                k = (j + 1) % seg_r
                new.append(self.bm.faces.new((a[j], b[j], b[k], a[k])))
        idx = self._mi(mat)
        for f in new:
            f.material_index = idx

    def flange(self, r_pipe, loc, axis, mat="MI_SteelDark", bolts=8):
        """Flange disc with bolt heads."""
        rf = r_pipe * 1.75
        self.cyl(rf, 0.05, loc, mat, axis=axis, seg=32)
        rb = r_pipe * 1.45
        d = Vector((0, 0, 1))
        d.rotate(AXIS_ROT[axis].to_3x3())
        for i in range(bolts):
            a = TAU * i / bolts
            off = Vector((math.cos(a) * rb, math.sin(a) * rb, 0))
            off.rotate(AXIS_ROT[axis].to_3x3())
            p = Vector(loc) + off
            self.cyl(0.018, 0.09, p, "MI_Galvanized", axis=axis, seg=6)

    def i_beam(self, length, depth, width, loc, axis, mat="MI_SteelDark", tf=0.02, tw=0.012):
        """Simple I/H section along an axis (X, Y or Z)."""
        x, y, z = loc
        if axis == "Z":
            self.box((width, tf, length), (x, y + depth / 2 - tf / 2, z), mat)
            self.box((width, tf, length), (x, y - depth / 2 + tf / 2, z), mat)
            self.box((tw, depth, length), (x, y, z), mat)
        elif axis == "X":
            self.box((length, width, tf), (x, y, z + depth / 2 - tf / 2), mat)
            self.box((length, width, tf), (x, y, z - depth / 2 + tf / 2), mat)
            self.box((length, tw, depth), (x, y, z), mat)
        else:
            self.box((width, length, tf), (x, y, z + depth / 2 - tf / 2), mat)
            self.box((width, length, tf), (x, y, z - depth / 2 + tf / 2), mat)
            self.box((tw, length, depth), (x, y, z), mat)

    def handrail_ring(self, R, z, posts=16):
        self.torus(R, 0.024, (0, 0, z + 1.05), "MI_SafetyYellow", seg_R=64, seg_r=8)
        self.torus(R, 0.02, (0, 0, z + 0.55), "MI_SafetyYellow", seg_R=64, seg_r=8)
        for i in range(posts):
            a = TAU * i / posts
            self.cyl(0.024, 1.05, (math.cos(a) * R, math.sin(a) * R, z + 0.525), "MI_SafetyYellow", seg=8)

    def ladder(self, x, y, z0, z1, axis_out="X"):
        h = z1 - z0
        for s in (-0.22, 0.22):
            self.cyl(0.025, h, (x, y + s, z0 + h / 2), "MI_SafetyYellow", seg=8)
        n = int(h / 0.3)
        for i in range(n):
            self.cyl(0.015, 0.44, (x, y, z0 + 0.3 * (i + 0.5)), "MI_SafetyYellow", axis="Y", seg=6)
        # safety cage hoops
        z = z0 + 2.2
        while z < z1:
            self.torus(0.38, 0.015, (x + 0.38, y, z), "MI_SafetyYellow", seg_R=24, seg_r=6)
            z += 1.5

    # ---- finalize -------------------------------------------------------
    def build_object(self):
        bm = self.bm
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        # sharp edges by angle, everything else smooth
        for f in bm.faces:
            f.smooth = True
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0) > math.radians(35):
                e.smooth = False
        # box-projected UVs (world-scale, consistent texel density)
        uv = bm.loops.layers.uv.new("UVMap")
        for f in bm.faces:
            n = f.normal
            ax = max(range(3), key=lambda i: abs(n[i]))
            for lp in f.loops:
                c = lp.vert.co
                if ax == 0:
                    lp[uv].uv = (c.y / UV_TILE_M, c.z / UV_TILE_M)
                elif ax == 1:
                    lp[uv].uv = (c.x / UV_TILE_M, c.z / UV_TILE_M)
                else:
                    lp[uv].uv = (c.x / UV_TILE_M, c.y / UV_TILE_M)
        me = bpy.data.meshes.new(self.name)
        bm.to_mesh(me)
        bm.free()
        for key in self.mats:
            me.materials.append(get_blender_material(key))
        obj = bpy.data.objects.new(self.name, me)
        bpy.context.scene.collection.objects.link(obj)
        return obj


_mat_cache = {}


def get_blender_material(key):
    """Preview material in Blender: real 2K textures + tint, same as Unreal."""
    if key in _mat_cache:
        return _mat_cache[key]
    spec = CFG["materials"][key]
    mat = bpy.data.materials.new(key)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tset = spec["set"]

    def tex(suffix, noncolor):
        p = os.path.join(OUT_TEX, f"T_{tset}_{suffix}.png")
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(p, check_existing=True)
        if noncolor:
            n.image.colorspace_settings.name = "Non-Color"
        return n

    bc, orm_n, nrm = tex("BaseColor", False), tex("ORM", True), tex("Normal", True)
    tint = nt.nodes.new("ShaderNodeRGB")
    tint.outputs[0].default_value = (*spec["tint"], 1.0)
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    nt.links.new(bc.outputs["Color"], mul.inputs[6])
    nt.links.new(tint.outputs[0], mul.inputs[7])
    nt.links.new(mul.outputs[2], bsdf.inputs["Base Color"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(orm_n.outputs["Color"], sep.inputs[0])
    nt.links.new(sep.outputs[1], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs[2], bsdf.inputs["Metallic"])
    # DirectX normal -> flip green for Blender preview
    sep2 = nt.nodes.new("ShaderNodeSeparateColor")
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    comb = nt.nodes.new("ShaderNodeCombineColor")
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nrm.outputs["Color"], sep2.inputs[0])
    nt.links.new(sep2.outputs[0], comb.inputs[0])
    nt.links.new(sep2.outputs[1], inv.inputs[1])
    nt.links.new(inv.outputs[0], comb.inputs[1])
    nt.links.new(sep2.outputs[2], comb.inputs[2])
    nt.links.new(comb.outputs[0], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    mat.diffuse_color = (*spec["tint"], 1.0)
    _mat_cache[key] = mat
    return mat


# =========================================================================== ASSETS (metres)

def build_pump():
    """End-suction centrifugal pump + TEFC motor on a skid. Shaft along +X, suction at -X."""
    a = Asset("SM_Pump_Centrifugal")
    # skid / baseplate with drip lip
    a.box((2.6, 1.0, 0.16), (0, 0, 0.08), "MI_SteelDark")
    a.box((2.6, 0.06, 0.06), (0, 0.47, 0.19), "MI_SteelDark")
    a.box((2.6, 0.06, 0.06), (0, -0.47, 0.19), "MI_SteelDark")
    # grout pad
    a.box((2.8, 1.2, 0.1), (0, 0, -0.04), "MI_Concrete")
    # pump pedestal + volute casing (disc around X axis, offset scroll)
    a.box((0.35, 0.5, 0.32), (-0.7, 0, 0.32), "MI_PumpBlue")
    a.cyl(0.40, 0.30, (-0.72, 0, 0.78), "MI_PumpBlue", axis="X", seg=48)
    a.cyl(0.30, 0.34, (-0.72, 0.04, 0.80), "MI_PumpBlue", axis="X", seg=48)
    a.torus(0.40, 0.04, (-0.72, 0, 0.78), "MI_PumpBlue", axis="X", seg_R=48, seg_r=10)
    # suction nozzle (horizontal -X) with flange
    a.cyl(0.13, 0.35, (-1.03, 0, 0.78), "MI_PumpBlue", axis="X")
    a.flange(0.13, (-1.22, 0, 0.78), "X")
    # discharge nozzle (vertical) with flange — hookup point (-0.7, 0, 1.1) m
    a.cyl(0.10, 0.28, (-0.70, 0, 1.0), "MI_PumpBlue")
    a.flange(0.10, (-0.70, 0, 1.12), "Z")
    # bearing frame + shaft
    a.cyl(0.16, 0.5, (-0.32, 0, 0.78), "MI_PumpBlue", axis="X")
    a.box((0.4, 0.35, 0.36), (-0.32, 0, 0.42), "MI_PumpBlue")
    a.cyl(0.04, 0.4, (0.05, 0, 0.78), "MI_Galvanized", axis="X", seg=16)
    # coupling guard (safety yellow, half-cylinder look)
    a.cyl(0.17, 0.42, (0.1, 0, 0.78), "MI_SafetyYellow", axis="X")
    a.box((0.42, 0.34, 0.25), (0.1, 0, 0.58), "MI_SafetyYellow")
    # motor: body with cooling fins, fan cowl, terminal box, feet
    a.cyl(0.32, 0.95, (0.85, 0, 0.78), "MI_MotorGrey", axis="X", seg=48)
    for i in range(14):
        a.cyl(0.355, 0.025, (0.45 + i * 0.062, 0, 0.78), "MI_MotorGrey", axis="X", seg=48)
    a.cyl(0.34, 0.28, (1.46, 0, 0.78), "MI_MotorGrey", axis="X", seg=48)
    a.cyl(0.30, 0.02, (1.61, 0, 0.78), "MI_SteelDark", axis="X", seg=48)
    a.box((0.26, 0.22, 0.2), (0.85, -0.3, 1.1), "MI_MotorGrey")
    a.cyl(0.03, 0.25, (0.85, -0.3, 1.32), "MI_Galvanized", seg=12)
    for x in (0.55, 1.15):
        a.box((0.14, 0.7, 0.32), (x, 0, 0.32), "MI_MotorGrey")
    # anchor bolts
    for x in (-1.1, -0.3, 0.55, 1.15):
        for y in (-0.42, 0.42):
            a.cyl(0.025, 0.08, (x, y, 0.2), "MI_Galvanized", seg=6)
    # vibration sensor puck on bearing housing (the ML hero detail)
    a.cyl(0.03, 0.06, (-0.32, 0.17, 0.92), "MI_SafetyYellow", axis="Y", seg=12)
    return a.build_object()


def build_exchanger():
    """Shell & tube exchanger (TEMA BEM-ish), horizontal along X, on two saddles."""
    a = Asset("SM_HeatExchanger")
    L, R, zc = 5.0, 0.5, 1.3
    a.cyl(R, L, (0, 0, zc), "MI_PipeGrey", axis="X", seg=48)
    # girth flanges + channel at -X, bonnet at +X
    for x in (-L / 2, L / 2):
        a.cyl(R + 0.08, 0.1, (x, 0, zc), "MI_SteelDark", axis="X", seg=48)
        for i in range(24):
            ang = TAU * i / 24
            a.cyl(0.02, 0.18, (x, math.cos(ang) * (R + 0.04), zc + math.sin(ang) * (R + 0.04)), "MI_Galvanized", axis="X", seg=6)
    a.cyl(R, 0.6, (-L / 2 - 0.35, 0, zc), "MI_PipeGrey", axis="X", seg=48)
    a.cyl(R + 0.08, 0.08, (-L / 2 - 0.68, 0, zc), "MI_SteelDark", axis="X", seg=48)
    a.sphere(R, (L / 2 + 0.05, 0, zc), "MI_PipeGrey", scale=(0.55, 1, 1), seg=48, rings=24)
    # saddles
    for x in (-1.5, 1.5):
        a.box((0.3, 1.1, 0.8), (x, 0, 0.4), "MI_SteelDark")
        a.box((0.5, 1.3, 0.05), (x, 0, 0.025), "MI_SteelDark")
        a.box((0.6, 1.4, 0.15), (x, 0, -0.05), "MI_Concrete")
    # nozzles: shell in/out (top/bottom), tube in/out on channel
    for (x, up) in ((-1.6, True), (1.7, False)):
        z0 = zc + (R + 0.25) if up else zc - (R + 0.15)
        a.cyl(0.12, 0.5 if up else 0.3, (x, 0, z0 - (0.0 if up else 0)), "MI_PipeGrey")
        a.flange(0.12, (x, 0, z0 + (0.25 if up else -0.15)), "Z")
    for z in (zc + 0.25, zc - 0.25):
        a.cyl(0.1, 0.45, (-L / 2 - 0.35, -0.55, z), "MI_PipeGrey", axis="Y")
        a.flange(0.1, (-L / 2 - 0.35, -0.78, z), "Y")
    # nameplate + temperature indicator wells
    a.box((0.3, 0.02, 0.18), (0, -R - 0.01, zc + 0.1), "MI_Galvanized")
    for x in (-1.0, 1.0):
        a.cyl(0.025, 0.2, (x, 0, zc + R + 0.08), "MI_Galvanized", seg=8)
        a.sphere(0.06, (x, 0, zc + R + 0.22), "MI_Galvanized", seg=16, rings=8)
    return a.build_object()


def build_column():
    """Atmospheric distillation column, ~42 m tall, swaged with platforms, ladders, overhead line."""
    a = Asset("SM_DistillationColumn")
    # foundation + skirt
    a.cyl(3.4, 0.6, (0, 0, 0.3), "MI_Concrete", seg=64)
    a.cyl(2.25, 4.0, (0, 0, 2.6), "MI_SteelDark", seg=64)
    a.box((0.9, 0.05, 1.4), (0, -2.24, 1.4), "MI_SteelDark")  # skirt access opening frame
    # bottom section (larger diameter), cone transition, top section
    a.cyl(2.2, 14.0, (0, 0, 11.6), "MI_Cladding", seg=64)
    a.cyl(2.2, 3.0, (0, 0, 20.1), "MI_Cladding", seg=64, r2=1.6)
    a.cyl(1.6, 19.0, (0, 0, 31.1), "MI_Cladding", seg=64)
    a.sphere(1.6, (0, 0, 40.6), "MI_Cladding", scale=(1, 1, 0.5), seg=64, rings=24)
    a.sphere(2.2, (0, 0, 4.6), "MI_Cladding", scale=(1, 1, 0.5), seg=64, rings=24)
    # insulation support rings / stiffeners
    for z in range(6, 19, 2):
        a.torus(2.22, 0.03, (0, 0, z), "MI_Galvanized", seg_R=64, seg_r=6)
    for z in range(23, 40, 2):
        a.torus(1.62, 0.03, (0, 0, z), "MI_Galvanized", seg_R=64, seg_r=6)
    # platforms (grating disc + toe plate + handrail)
    for z, r in ((8.0, 2.2), (14.0, 2.2), (24.0, 1.6), (30.0, 1.6), (36.0, 1.6), (41.2, 1.6)):
        R = r + 1.3
        a.cyl(R, 0.06, (0, 0, z), "MI_Grating", seg=64)
        a.cyl(R, 0.15, (0, 0, z + 0.1), "MI_SafetyYellow", seg=64, caps=False)
        a.handrail_ring(R - 0.05, z)
        # platform brackets
        for i in range(8):
            ang = TAU * i / 8
            c, s = math.cos(ang), math.sin(ang)
            mid = r + 0.65
            a.box((1.3, 0.08, 0.25), (c * mid, s * mid, z - 0.15), "MI_SteelDark", rot_z=ang)
    # ladder up the +X side between platforms
    a.ladder(2.6, 0.9, 0.6, 14.1)
    a.ladder(2.0, 0.9, 14.1, 41.3)
    # nozzles: feed, draws, manways
    for z, side in ((6.5, 1), (12.5, -1), (18.0, 1)):
        a.cyl(0.35, 0.8, (side * 2.55, 0, z), "MI_Cladding", axis="X")
        a.flange(0.35, (side * 2.95, 0, z), "X")
    for z in (10.0, 26.0, 33.0):
        r = 2.2 if z < 20 else 1.6
        a.cyl(0.45, 0.5, (0, -(r + 0.2), z), "MI_SteelDark", axis="Y")
        a.cyl(0.6, 0.12, (0, -(r + 0.47), z), "MI_SteelDark", axis="Y")
        a.cyl(0.07, 0.4, (0.5, -(r + 0.55), z), "MI_Galvanized", seg=8)  # davit
    # overhead vapour line: up from head, bends over and down the side
    a.cyl(0.45, 2.5, (0, 0, 42.2), "MI_Cladding")
    # 180-degree goose-neck in the XZ plane (torus on Y axis), then downcomer at x = 3.0
    a.torus(1.5, 0.45, (1.5, 0, 43.45), "MI_Cladding", axis="Y", seg_R=32, seg_r=24, arc=math.pi, start=math.pi)
    a.cyl(0.45, 41.0, (3.0, 0, 22.95), "MI_Cladding")
    # pipe supports clamping the downcomer to the column
    for z in (12, 22, 32):
        if z < 20:
            a.box((0.8, 0.12, 0.12), (2.6, 0, z), "MI_SteelDark")
        else:
            a.box((1.4, 0.12, 0.12), (2.3, 0, z), "MI_SteelDark")
    return a.build_object()


def build_furnace():
    """Box-type crude charge heater: radiant box, convection section, stack, burners, peepholes."""
    a = Asset("SM_Furnace")
    W, D, H = 9.0, 6.0, 11.0
    # legs + floor
    for x in (-W / 2 + 0.3, 0, W / 2 - 0.3):
        for y in (-D / 2 + 0.3, D / 2 - 0.3):
            a.i_beam(2.5, 0.35, 0.35, (x, y, 1.25), "Z")
            a.cyl(0.45, 0.4, (x, y, 0.2), "MI_Concrete", seg=24)
    a.box((W, D, 0.3), (0, 0, 2.65), "MI_SteelDark")
    # radiant box casing + vertical stiffeners + horizontal bands
    a.box((W, D, H), (0, 0, 2.8 + H / 2), "MI_Refractory")
    for i in range(10):
        x = -W / 2 + W * i / 9
        for y in (-D / 2 - 0.08, D / 2 + 0.08):
            a.box((0.18, 0.16, H), (x, y, 2.8 + H / 2), "MI_SteelDark")
    for z in (5.5, 9.0, 12.5):
        a.box((W + 0.3, 0.12, 0.18), (0, -D / 2 - 0.12, z), "MI_SteelDark")
        a.box((W + 0.3, 0.12, 0.18), (0, D / 2 + 0.12, z), "MI_SteelDark")
    # peep doors
    for x in (-3, 0, 3):
        for z in (4.5, 8.0):
            a.box((0.35, 0.1, 0.35), (x, -D / 2 - 0.15, z), "MI_SteelDark")
    # hopper transition + convection section + breeching + stack
    zt = 2.8 + H
    a.box((W * 0.75, D * 0.75, 1.0), (0, 0, zt + 0.5), "MI_Refractory")  # stepped hopper
    a.box((W * 0.55, D * 0.55, 1.0), (0, 0, zt + 1.5), "MI_Refractory")
    a.box((4.0, 2.6, 4.0), (0, 0, zt + 4.0), "MI_Refractory")
    for z in (zt + 2.5, zt + 4.0, zt + 5.5):
        a.box((4.3, 2.9, 0.15), (0, 0, z), "MI_SteelDark")
    a.cyl(1.1, 2.0, (0, 0, zt + 7.0), "MI_Refractory", seg=48, r2=0.95)
    a.cyl(0.95, 22.0, (0, 0, zt + 19.0), "MI_SteelDark", seg=48)
    for z in (zt + 26.0, zt + 28.5):  # aviation bands
        a.cyl(0.97, 1.2, (0, 0, z), "MI_PipeRed", seg=48)
    a.cyl(0.97, 1.2, (0, 0, zt + 27.25), "MI_TankWhite", seg=48)
    a.cyl(1.05, 0.25, (0, 0, zt + 30.1), "MI_SteelDark", seg=48)
    # stack platform
    a.cyl(2.0, 0.06, (0, 0, zt + 24.0), "MI_Grating", seg=48)
    a.handrail_ring(1.95, zt + 24.0, posts=12)
    # burners under floor
    for x in (-3.0, -1.0, 1.0, 3.0):
        for y in (-1.2, 1.2):
            a.cyl(0.3, 0.9, (x, y, 2.05), "MI_SteelDark", seg=24)
            a.cyl(0.06, 0.6, (x + 0.35, y, 1.9), "MI_PipeYellow", axis="X", seg=8)  # fuel gas
    # crossover pipes from convection to radiant (coil inlet/outlet)
    for y in (-0.8, 0.8):
        a.cyl(0.15, 3.0, (2.4, y, zt + 2.0), "MI_PipeGrey", axis="X")
    # side platform + ladder
    a.box((W, 1.2, 0.06), (0, -D / 2 - 0.9, 8.0), "MI_Grating")
    a.ladder(W / 2 + 0.6, -D / 2 - 0.9, 0.0, 8.05)
    return a.build_object()


def build_tank():
    """API 650 cone-roof storage tank, 20 m dia x 13 m, with wind girder, roof handrail, stair."""
    a = Asset("SM_StorageTank")
    R, H = 10.0, 13.0
    a.cyl(R + 0.6, 0.4, (0, 0, 0.2), "MI_Concrete", seg=96)  # ringwall
    a.cyl(R, H, (0, 0, 0.4 + H / 2), "MI_TankWhite", seg=96)
    for i in range(1, 6):  # shell course weld lines
        a.torus(R + 0.005, 0.015, (0, 0, 0.4 + i * H / 6), "MI_TankWhite", seg_R=96, seg_r=4)
    a.cyl(R + 0.1, 1.2, (0, 0, H + 0.4 + 0.6), "MI_TankWhite", seg=96, r2=0.6)  # cone roof
    a.torus(R + 0.15, 0.08, (0, 0, H + 0.4), "MI_SteelDark", seg_R=96, seg_r=8)
    a.torus(R + 0.4, 0.06, (0, 0, H * 0.75), "MI_SteelDark", seg_R=96, seg_r=6)  # wind girder
    # roof edge handrail (partial arc near stair)
    a.torus(R - 0.1, 0.025, (0, 0, H + 1.5), "MI_SafetyYellow", seg_R=64, seg_r=6, arc=math.radians(70))
    for i in range(8):
        ang = math.radians(70) * i / 7
        a.cyl(0.025, 1.0, (math.cos(ang) * (R - 0.1), math.sin(ang) * (R - 0.1), H + 1.0), "MI_SafetyYellow", seg=6)
    # spiral stair approximated with treads around the shell
    steps = 52
    for i in range(steps):
        ang = math.radians(-60) + math.radians(130) * i / steps
        z = 0.6 + H * i / steps
        c, s = math.cos(ang), math.sin(ang)
        a.box((0.9, 0.3, 0.04), (c * (R + 0.55), s * (R + 0.55), z), "MI_Grating", rot_z=ang)
        if i % 4 == 0:
            a.cyl(0.02, 1.0, (c * (R + 1.0), s * (R + 1.0), z + 0.5), "MI_SafetyYellow", seg=6)
    # nozzles + manway
    for ang, z in ((math.radians(200), 0.9), (math.radians(160), 0.9), (math.radians(180), 1.4)):
        c, s = math.cos(ang), math.sin(ang)
        a.cyl(0.3, 0.8, (c * (R + 0.4), s * (R + 0.4), z), "MI_TankWhite", axis="X" if abs(c) > 0.7 else "Y")
    a.cyl(0.35, 0.3, (0, 0, H + 1.65), "MI_SteelDark", seg=24)  # roof vent
    a.sphere(0.4, (0, 0, H + 1.9), "MI_SteelDark", scale=(1, 1, 0.5), seg=24, rings=8)
    # level gauge board
    a.box((0.15, 0.08, H - 1.0), (-(R + 0.12), 0, 0.4 + H / 2), "MI_SteelDark")
    return a.build_object()


def build_pipe_rack():
    """One 18 m segment of a 2-level pipe rack, origin at x=0 start, centred on Y, width 6 m."""
    a = Asset("SM_PipeRack")
    W, levels = 6.0, (5.0, 8.0)
    for x in (0.0, 6.0, 12.0):
        for y in (-W / 2, W / 2):
            a.box((0.8, 0.8, 0.6), (x, y, 0.3), "MI_Concrete")
            a.i_beam(8.6, 0.35, 0.35, (x, y, 0.6 + 4.3), "Z")
            a.box((0.5, 0.5, 0.03), (x, y, 0.615), "MI_SteelDark")
        for z in levels:
            a.i_beam(W, 0.4, 0.2, (x, 0, z - 0.2), "Y")
        # knee braces
        for y, sgn in ((-W / 2, 1), (W / 2, -1)):
            a.box((0.12, 1.2, 0.12), (x, y + sgn * 0.5, levels[0] - 0.75), "MI_SteelDark")
    # longitudinal struts
    for y in (-W / 2, W / 2):
        for z in levels:
            a.i_beam(18.0, 0.3, 0.15, (9.0, y, z - 0.15), "X")
    # pipes: (y, radius, material) per level, length 18 m along X
    lower = [(-2.4, 0.30, "MI_PipeGrey"), (-1.6, 0.20, "MI_PipeGreen"), (-0.9, 0.25, "MI_PipeGrey"),
             (-0.2, 0.15, "MI_PipeYellow"), (0.4, 0.35, "MI_Cladding"), (1.3, 0.20, "MI_PipeRed"),
             (2.0, 0.15, "MI_PipeGrey"), (2.6, 0.10, "MI_PipeGreen")]
    upper = [(-2.3, 0.15, "MI_PipeGrey"), (-1.5, 0.40, "MI_Cladding"), (-0.4, 0.20, "MI_PipeYellow"),
             (0.4, 0.25, "MI_PipeGrey"), (1.2, 0.10, "MI_PipeGreen"), (1.8, 0.10, "MI_PipeGrey"),
             (2.4, 0.30, "MI_Cladding")]
    for z, pipes in ((levels[0], lower), (levels[1], upper)):
        for y, r, m in pipes:
            a.cyl(r, 18.0, (9.0, y, z + r + 0.01), m, axis="X", seg=24 if r < 0.2 else 32)
            # pipe shoes on each beam
            for x in (0.0, 6.0, 12.0):
                a.box((0.3, r * 1.2, 0.08), (x, y, z + 0.04), "MI_SteelDark")
    # cable tray on top level
    a.box((18.0, 0.6, 0.02), (9.0, 0, levels[1] + 1.2), "MI_Galvanized")
    for y in (-0.3, 0.3):
        a.box((18.0, 0.02, 0.12), (9.0, y, levels[1] + 1.26), "MI_Galvanized")
    for x in (0.0, 6.0, 12.0):
        a.box((0.06, 0.06, 1.2), (x, 0, levels[1] + 0.6), "MI_SteelDark")
    return a.build_object()


def build_valve():
    """Flanged gate valve, flow along X, rising stem with yoke & handwheel. ~1.4 m tall."""
    a = Asset("SM_GateValve")
    a.sphere(0.18, (0, 0, 0), "MI_PipeGrey", scale=(1.1, 0.9, 1.3), seg=32, rings=16)
    for x in (-0.22, 0.22):
        a.cyl(0.12, 0.2, (x * 0.8, 0, 0), "MI_PipeGrey", axis="X")
        a.flange(0.12, (x, 0, 0), "X")
    a.cyl(0.12, 0.35, (0, 0, 0.35), "MI_PipeGrey", seg=24)       # bonnet
    a.cyl(0.18, 0.05, (0, 0, 0.55), "MI_SteelDark", seg=24)      # bonnet flange
    for y in (-0.1, 0.1):
        a.box((0.04, 0.03, 0.45), (0, y, 0.8), "MI_SteelDark")   # yoke arms
    a.box((0.1, 0.26, 0.05), (0, 0, 1.03), "MI_SteelDark")
    a.cyl(0.02, 0.5, (0, 0, 1.05), "MI_Galvanized", seg=8)       # stem
    a.torus(0.22, 0.02, (0, 0, 1.12), "MI_PipeRed", seg_R=32, seg_r=8)
    for i in range(4):
        ang = TAU * i / 4
        a.box((0.22, 0.02, 0.02), (math.cos(ang) * 0.11, math.sin(ang) * 0.11, 1.12), "MI_PipeRed", rot_z=ang)
    a.cyl(0.04, 0.06, (0, 0, 1.12), "MI_PipeRed", seg=12)
    return a.build_object()


def build_pipe():
    """Unit pipe: radius 0.15 m, 1 m long along Z, centred at origin (scale Z in Unreal)."""
    a = Asset("SM_Pipe_Straight")
    a.cyl(0.15, 1.0, (0, 0, 0), "MI_PipeGrey", seg=32)
    return a.build_object()


def build_ground():
    """20 x 20 m concrete paving slab, top surface at z=0."""
    a = Asset("SM_GroundTile")
    a.box((20.0, 20.0, 0.3), (0, 0, -0.15), "MI_Concrete")
    a.box((20.0, 0.25, 0.04), (0, -9.9, 0.0), "MI_SafetyYellow")  # painted walkway line
    return a.build_object()


# =========================================================================== EXPORT

def export_fbx(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    path = os.path.join(OUT_MESH, obj.name + ".fbx")
    bpy.ops.export_scene.fbx(
        filepath=path, use_selection=True, object_types={"MESH"},
        apply_scale_options="FBX_SCALE_UNITS", apply_unit_scale=True, global_scale=1.0,
        mesh_smooth_type="EDGE", use_mesh_modifiers=True, add_leaf_bones=False,
        bake_anim=False, path_mode="STRIP", embed_textures=False,
    )
    # glTF for the web viewer: geometry + UVs + material slot names (textures are shared PNGs)
    glb = os.path.join(OUT_WEB, obj.name + ".glb")
    kw = dict(filepath=glb, export_format="GLB", use_selection=True, export_apply=True, export_yup=True)
    try:
        bpy.ops.export_scene.gltf(export_materials="EXPORT", export_image_format="NONE", **kw)
    except TypeError:
        bpy.ops.export_scene.gltf(export_materials="EXPORT", **kw)
    tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    log(f"exported {obj.name}.fbx  (~{tris:,} tris, slots: {[m.name for m in obj.data.materials]})")


def main():
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0

    log(f"generating tileable PBR textures at {TEX_RES}x{TEX_RES} (cap 2048)")
    make_texture_sets(TEX_RES)

    builders = [build_pump, build_exchanger, build_column, build_furnace, build_tank,
                build_pipe_rack, build_valve, build_pipe, build_ground]
    objs = []
    for b in builders:
        obj = b()
        export_fbx(obj)
        objs.append(obj)

    # lay out a showroom row in the .blend for easy inspection / hand editing
    x = 0.0
    for obj in objs:
        dims = obj.dimensions
        obj.location.x = x + dims.x / 2
        x += dims.x + 4.0
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_BLEND, "RefineryAssets.blend"))
    log(f"done in {time.time() - t0:.1f}s -> {OUT_MESH}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
