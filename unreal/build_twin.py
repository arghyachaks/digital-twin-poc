"""
Refinery Digital Twin - Unreal Engine 5 builder
===============================================
Runs inside the UE5 editor (Python Editor Script Plugin):
    UnrealEditor.exe RefineryTwin.uproject -ExecutePythonScript="<root>/unreal/build_twin.py"

Steps
  1. Import 2K textures from Content_Source/Textures (MaxTextureSize forced to 2048, correct sRGB/compression)
  2. Build master material M_TwinMaster (BaseColor*Tint, ORM, Normal, UVScale, HealthColor/HealthGlow emissive)
  3. Create one Material Instance per entry in config/materials.json
  4. Import FBX meshes from Content_Source/Meshes, enable Nanite, bind slots -> instances by slot name
  5. Build /Game/DigitalTwin/Maps/L_Refinery from config/plant_layout.json:
     ground, equipment (tagged with asset IDs), pipe racks, pump hookups, labels, Lumen sky/sun/fog/clouds
Re-runnable: assets are replaced/updated, the level is cleared and rebuilt.
"""
import json
import math
import os

import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC_TEX = os.path.join(ROOT, "Content_Source", "Textures")
SRC_MESH = os.path.join(ROOT, "Content_Source", "Meshes")
MATS = json.load(open(os.path.join(ROOT, "config", "materials.json")))
LAYOUT = json.load(open(os.path.join(ROOT, "config", "plant_layout.json")))
MAX_TEX = min(int(MATS.get("max_texture_size", 2048)), 2048)

G = "/Game/DigitalTwin"
P_TEX, P_MAT, P_MESH, P_MAP = f"{G}/Textures", f"{G}/Materials", f"{G}/Meshes", f"{G}/Maps"
LEVEL = f"{P_MAP}/L_Refinery"

asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary


def log(m):
    unreal.log(f"[TwinBuilder] {m}")


# --------------------------------------------------------------------------- import helpers
def run_import(files, dest, options=None):
    tasks = []
    for f in files:
        t = unreal.AssetImportTask()
        t.set_editor_property("filename", f)
        t.set_editor_property("destination_path", dest)
        t.set_editor_property("automated", True)
        t.set_editor_property("replace_existing", True)
        t.set_editor_property("save", True)
        if options is not None:
            t.set_editor_property("options", options)
        tasks.append(t)
    asset_tools.import_asset_tasks(tasks)


def import_textures():
    files = sorted(os.path.join(SRC_TEX, f) for f in os.listdir(SRC_TEX) if f.lower().endswith(".png"))
    run_import(files, P_TEX)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        tex = unreal.load_asset(f"{P_TEX}/{name}")
        if tex is None:
            unreal.log_warning(f"texture failed: {name}")
            continue
        tex.set_editor_property("max_texture_size", MAX_TEX)  # hard 2K cap
        if name.endswith("_Normal"):
            tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
            tex.set_editor_property("srgb", False)
        elif name.endswith("_ORM"):
            tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
            tex.set_editor_property("srgb", False)
        else:
            tex.set_editor_property("srgb", True)
        EAL.save_loaded_asset(tex)
    log(f"imported {len(files)} textures (max {MAX_TEX}px)")


def tex(set_name, kind):
    return unreal.load_asset(f"{P_TEX}/T_{set_name}_{kind}")


# --------------------------------------------------------------------------- materials
def get_or_create(name, path, cls, factory):
    full = f"{path}/{name}"
    if EAL.does_asset_exist(full):
        return unreal.load_asset(full)
    return asset_tools.create_asset(name, path, cls, factory)


def build_master_material():
    mat = get_or_create("M_TwinMaster", P_MAT, unreal.Material, unreal.MaterialFactoryNew())
    MEL.delete_all_material_expressions(mat)
    X = MEL.create_material_expression

    tc = X(mat, unreal.MaterialExpressionTextureCoordinate, -1400, 0)
    uvs = X(mat, unreal.MaterialExpressionScalarParameter, -1400, 120)
    uvs.set_editor_property("parameter_name", "UVScale")
    uvs.set_editor_property("default_value", 1.0)
    uvmul = X(mat, unreal.MaterialExpressionMultiply, -1150, 40)
    MEL.connect_material_expressions(tc, "", uvmul, "A")
    MEL.connect_material_expressions(uvs, "", uvmul, "B")

    def sampler(pname, default, stype, y):
        s = X(mat, unreal.MaterialExpressionTextureSampleParameter2D, -900, y)
        s.set_editor_property("parameter_name", pname)
        s.set_editor_property("texture", default)
        s.set_editor_property("sampler_type", stype)
        MEL.connect_material_expressions(uvmul, "", s, "UVs")
        return s

    bc = sampler("BaseColor", tex("PaintedSteel", "BaseColor"), unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -300)
    ormt = sampler("ORM", tex("PaintedSteel", "ORM"), unreal.MaterialSamplerType.SAMPLERTYPE_MASKS, 50)
    nrm = sampler("Normal", tex("PaintedSteel", "Normal"), unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, 400)

    tint = X(mat, unreal.MaterialExpressionVectorParameter, -600, -450)
    tint.set_editor_property("parameter_name", "Tint")
    tint.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
    bcm = X(mat, unreal.MaterialExpressionMultiply, -350, -300)
    MEL.connect_material_expressions(bc, "RGB", bcm, "A")
    MEL.connect_material_expressions(tint, "", bcm, "B")
    MEL.connect_material_property(bcm, "", unreal.MaterialProperty.MP_BASE_COLOR)

    rs = X(mat, unreal.MaterialExpressionScalarParameter, -600, 180)
    rs.set_editor_property("parameter_name", "RoughnessScale")
    rs.set_editor_property("default_value", 1.0)
    rm = X(mat, unreal.MaterialExpressionMultiply, -350, 120)
    MEL.connect_material_expressions(ormt, "G", rm, "A")
    MEL.connect_material_expressions(rs, "", rm, "B")
    MEL.connect_material_property(rm, "", unreal.MaterialProperty.MP_ROUGHNESS)
    MEL.connect_material_property(ormt, "B", unreal.MaterialProperty.MP_METALLIC)
    MEL.connect_material_property(ormt, "R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    MEL.connect_material_property(nrm, "RGB", unreal.MaterialProperty.MP_NORMAL)

    # Health overlay driven at runtime by the twin (dynamic material instance):
    # emissive = HealthColor * HealthGlow * (0.65 + 0.35 * sin(time * PulseSpeed))
    hc = X(mat, unreal.MaterialExpressionVectorParameter, -900, 750)
    hc.set_editor_property("parameter_name", "HealthColor")
    hc.set_editor_property("default_value", unreal.LinearColor(0.0, 1.0, 0.2, 1))
    hg = X(mat, unreal.MaterialExpressionScalarParameter, -900, 900)
    hg.set_editor_property("parameter_name", "HealthGlow")
    hg.set_editor_property("default_value", 0.0)
    ps = X(mat, unreal.MaterialExpressionScalarParameter, -1150, 1050)
    ps.set_editor_property("parameter_name", "PulseSpeed")
    ps.set_editor_property("default_value", 4.0)
    tm = X(mat, unreal.MaterialExpressionTime, -1150, 950)
    tmul = X(mat, unreal.MaterialExpressionMultiply, -900, 1000)
    MEL.connect_material_expressions(tm, "", tmul, "A")
    MEL.connect_material_expressions(ps, "", tmul, "B")
    sn = X(mat, unreal.MaterialExpressionSine, -700, 1000)
    MEL.connect_material_expressions(tmul, "", sn, "")
    pulse = X(mat, unreal.MaterialExpressionMultiply, -550, 1000)
    pulse.set_editor_property("const_b", 0.35)
    MEL.connect_material_expressions(sn, "", pulse, "A")
    padd = X(mat, unreal.MaterialExpressionAdd, -400, 1000)
    padd.set_editor_property("const_b", 0.65)
    MEL.connect_material_expressions(pulse, "", padd, "A")
    e1 = X(mat, unreal.MaterialExpressionMultiply, -600, 800)
    MEL.connect_material_expressions(hc, "", e1, "A")
    MEL.connect_material_expressions(hg, "", e1, "B")
    e2 = X(mat, unreal.MaterialExpressionMultiply, -250, 850)
    MEL.connect_material_expressions(e1, "", e2, "A")
    MEL.connect_material_expressions(padd, "", e2, "B")
    MEL.connect_material_property(e2, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    try:
        mat.set_editor_property("used_with_nanite", True)
    except Exception:
        pass
    MEL.recompile_material(mat)
    EAL.save_loaded_asset(mat)
    log("built M_TwinMaster")
    return mat


def build_instances(master):
    out = {}
    for name, spec in MATS["materials"].items():
        mi = get_or_create(name, P_MAT, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        MEL.set_material_instance_parent(mi, master)
        s = spec["set"]
        MEL.set_material_instance_texture_parameter_value(mi, "BaseColor", tex(s, "BaseColor"))
        MEL.set_material_instance_texture_parameter_value(mi, "ORM", tex(s, "ORM"))
        MEL.set_material_instance_texture_parameter_value(mi, "Normal", tex(s, "Normal"))
        r, g, b = spec["tint"]
        MEL.set_material_instance_vector_parameter_value(mi, "Tint", unreal.LinearColor(r, g, b, 1))
        MEL.set_material_instance_scalar_parameter_value(mi, "UVScale", float(spec.get("uv_scale", 1.0)))
        MEL.update_material_instance(mi)
        EAL.save_loaded_asset(mi)
        out[name] = mi
    log(f"built {len(out)} material instances")
    return out


# --------------------------------------------------------------------------- meshes
def fbx_options():
    o = unreal.FbxImportUI()
    o.set_editor_property("import_mesh", True)
    o.set_editor_property("import_as_skeletal", False)
    o.set_editor_property("import_materials", False)
    o.set_editor_property("import_textures", False)
    o.set_editor_property("import_animations", False)
    o.set_editor_property("automated_import_should_detect_type", False)
    o.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_STATIC_MESH)
    sm = o.get_editor_property("static_mesh_import_data")
    sm.set_editor_property("combine_meshes", True)
    sm.set_editor_property("auto_generate_collision", True)
    sm.set_editor_property("generate_lightmap_u_vs", False)
    sm.set_editor_property("normal_import_method", unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS)
    try:
        sm.set_editor_property("build_nanite", True)
    except Exception:
        pass
    return o


def import_meshes(instances):
    # Prefer the legacy FBX path so FbxImportUI options apply (UE 5.5+ defaults to Interchange)
    for cmd in ("Interchange.FeatureFlags.Import.FBX 0",):
        unreal.SystemLibrary.execute_console_command(None, cmd)
    files = sorted(os.path.join(SRC_MESH, f) for f in os.listdir(SRC_MESH) if f.lower().endswith(".fbx"))
    run_import(files, P_MESH, fbx_options())
    meshes = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        mesh = unreal.load_asset(f"{P_MESH}/{name}")
        if mesh is None:
            unreal.log_warning(f"mesh failed: {name}")
            continue
        try:
            ns = mesh.get_editor_property("nanite_settings")
            ns.set_editor_property("enabled", True)
            mesh.set_editor_property("nanite_settings", ns)
        except Exception as e:
            unreal.log_warning(f"nanite not set on {name}: {e}")
        # bind material slots by name (Blender slot name == material instance name)
        for i, slot in enumerate(mesh.get_editor_property("static_materials")):
            slot_name = str(slot.get_editor_property("material_slot_name"))
            key = next((k for k in instances if slot_name == k or slot_name.startswith(k)), None)
            if key:
                mesh.set_material(i, instances[key])
            else:
                unreal.log_warning(f"{name}: no instance for slot '{slot_name}'")
        EAL.save_loaded_asset(mesh)
        meshes[name] = mesh
    log(f"imported {len(meshes)} meshes with Nanite")
    return meshes


# --------------------------------------------------------------------------- level
EAS = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
LES = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)


def V(x, y, z):
    return unreal.Vector(float(x), float(y), float(z))


def R(pitch=0.0, yaw=0.0, roll=0.0):
    return unreal.Rotator(roll=float(roll), pitch=float(pitch), yaw=float(yaw))


def spawn_mesh(mesh, loc, rot=None, scale=None, label=None, tags=None, folder=None):
    a = EAS.spawn_actor_from_object(mesh, loc, rot or R())
    if scale is not None:
        a.set_actor_scale3d(scale)
    if label:
        a.set_actor_label(label)
    if tags:
        a.set_editor_property("tags", [unreal.Name(t) for t in tags])
    if folder:
        a.set_folder_path(folder)
    return a


def open_fresh_level():
    if EAL.does_asset_exist(LEVEL):
        EAL.delete_asset(LEVEL)
    LES.new_level(LEVEL)


def build_environment():
    sun = EAS.spawn_actor_from_class(unreal.DirectionalLight, V(0, 0, 5000), R(pitch=-32, yaw=-135))
    sun.set_actor_label("Sun")
    lc = sun.get_component_by_class(unreal.DirectionalLightComponent)
    lc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    lc.set_editor_property("intensity", 10.0)
    lc.set_editor_property("atmosphere_sun_light", True)
    lc.set_editor_property("light_source_angle", 0.8)
    lc.set_editor_property("temperature", 5600.0)
    lc.set_editor_property("use_temperature", True)

    EAS.spawn_actor_from_class(unreal.SkyAtmosphere, V(0, 0, 0)).set_actor_label("SkyAtmosphere")
    sky = EAS.spawn_actor_from_class(unreal.SkyLight, V(0, 0, 1000))
    sky.set_actor_label("SkyLight")
    sc = sky.get_component_by_class(unreal.SkyLightComponent)
    sc.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
    sc.set_editor_property("real_time_capture", True)

    fog = EAS.spawn_actor_from_class(unreal.ExponentialHeightFog, V(0, 0, 0))
    fog.set_actor_label("HeightFog")
    fc = fog.get_component_by_class(unreal.ExponentialHeightFogComponent)
    fc.set_editor_property("fog_density", 0.008)
    fc.set_editor_property("fog_height_falloff", 0.15)
    fc.set_editor_property("enable_volumetric_fog", True)

    try:
        EAS.spawn_actor_from_class(unreal.VolumetricCloud, V(0, 0, 0)).set_actor_label("VolumetricClouds")
    except Exception:
        pass

    ppv = EAS.spawn_actor_from_class(unreal.PostProcessVolume, V(0, 0, 0))
    ppv.set_actor_label("PostProcess_Global")
    ppv.set_editor_property("unbound", True)
    s = ppv.get_editor_property("settings")
    for k, v in (("override_bloom_intensity", True), ("bloom_intensity", 0.6),
                 ("override_auto_exposure_bias", True), ("auto_exposure_bias", 0.3),
                 ("override_vignette_intensity", True), ("vignette_intensity", 0.35),
                 ("override_film_grain_intensity", True), ("film_grain_intensity", 0.05),
                 ("override_ambient_occlusion_intensity", True), ("ambient_occlusion_intensity", 0.6)):
        try:
            s.set_editor_property(k, v)
        except Exception:
            pass
    ppv.set_editor_property("settings", s)

    for cls, lbl in ((unreal.PlayerStart, "PlayerStart"),):
        EAS.spawn_actor_from_class(cls, V(-6000, -9000, 300), R(yaw=55)).set_actor_label(lbl)


def label(text, loc, color=unreal.Color(255, 200, 40, 255), size=90):
    t = EAS.spawn_actor_from_class(unreal.TextRenderActor, loc, R(yaw=90))
    t.set_actor_label(f"Label_{text}")
    c = t.get_component_by_class(unreal.TextRenderComponent)
    c.set_text(text)
    c.set_editor_property("world_size", size)
    c.set_editor_property("horizontal_alignment", unreal.HorizTextAligment.EHTA_CENTER)
    c.set_editor_property("text_render_color", color)
    t.set_folder_path("Labels")
    t.set_editor_property("tags", [unreal.Name("TwinLabel"), unreal.Name(text)])
    return t


def build_layout(meshes):
    g = LAYOUT["ground"]
    gt = meshes.get(g["mesh"])
    if gt:
        ox, oy, oz = g["origin"]
        s = g["tile_size_cm"]
        for i in range(g["nx"]):
            for j in range(g["ny"]):
                spawn_mesh(gt, V(ox + s * (i + 0.5), oy + s * (j + 0.5), oz), label=f"Ground_{i}_{j}", folder="Environment/Ground")

    # equipment, tagged for data binding
    for e in LAYOUT["equipment"]:
        m = meshes.get(e["mesh"])
        if not m:
            continue
        x, y, z = e["loc"]
        a = spawn_mesh(m, V(x, y, z), R(yaw=e.get("yaw", 0)), label=f"{e['tag']}",
                       tags=["TwinAsset", e["tag"], e["mesh"]] + (["Hero"] if e.get("hero") else []),
                       folder=f"Equipment/{e['tag'][0]}")
        h = a.get_actor_bounds(False)[1].z * 2  # bounds extent -> height
        label(e["tag"], V(x, y - 150, z + h + 150), size=120 if e["mesh"] != "SM_Pump_Centrifugal" else 60)

    # pipe racks
    pr = LAYOUT["pipe_racks"]
    rack = meshes.get(pr["mesh"])
    if rack:
        for i in range(pr["count"]):
            spawn_mesh(rack, V(pr["x_start"] + i * pr["segment_cm"], pr["y"], 0), label=f"PipeRack_{i:02d}",
                       tags=["TwinStructure"], folder="Structures/PipeRack")

    # pump hookups: discharge valve + riser to rack + run to rack edge
    hk = LAYOUT["pump_hookups"]
    valve, pipe = meshes.get(hk["valve_mesh"]), meshes.get(hk["pipe_mesh"])
    rack_edge_y = pr["y"] + 300  # rack is 6 m wide centred on y
    if valve and pipe:
        dx, dy, dz = hk["discharge_offset"]
        top = hk["rack_pipe_z"]
        for e in LAYOUT["equipment"]:
            if e["mesh"] != "SM_Pump_Centrifugal":
                continue
            x, y, z = e["loc"][0] + dx, e["loc"][1] + dy, e["loc"][2] + dz
            v_z = z + 60
            spawn_mesh(valve, V(x, y, v_z), R(pitch=90), label=f"XV-{e['tag']}",
                       tags=["TwinAsset", f"XV-{e['tag']}", "Valve"], folder="Equipment/Valves")
            riser_from = v_z + 30
            spawn_mesh(pipe, V(x, y, (riser_from + top) / 2), scale=V(1, 1, (top - riser_from) / 100.0),
                       label=f"Riser_{e['tag']}", folder="Piping")
            run = y - rack_edge_y
            spawn_mesh(pipe, V(x, (y + rack_edge_y) / 2, top), R(roll=90), scale=V(1, 1, abs(run) / 100.0),
                       label=f"Run_{e['tag']}", folder="Piping")

    build_environment()
    unit = LAYOUT.get("unit", "CDU")
    label(f"{unit}  ·  CRUDE DISTILLATION UNIT", V(-2000, -6200, 1400), unreal.Color(255, 255, 255, 255), 260)


def frame_camera():
    try:
        ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
        ues.set_level_viewport_camera_info(V(-9500, -11000, 3200), R(pitch=-14, yaw=48))
    except Exception:
        pass
    cam = EAS.spawn_actor_from_class(unreal.CineCameraActor, V(-9500, -11000, 3200), R(pitch=-14, yaw=48))
    cam.set_actor_label("Cam_Hero_Overview")
    cam2 = EAS.spawn_actor_from_class(unreal.CineCameraActor, V(-2300, -3400, 260), R(pitch=-10, yaw=60))
    cam2.set_actor_label("Cam_Hero_P101A")


def main():
    # Release references held by a previously built level before re-importing assets (avoids GC assert)
    unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
    # UE 5.8 asserts when FBX "replace existing" hits a loaded mesh, so start from a clean slate
    for path in (LEVEL, P_MESH):
        if EAL.does_asset_exist(path) or EAL.does_directory_exist(path):
            (EAL.delete_asset if path == LEVEL else EAL.delete_directory)(path)
    unreal.SystemLibrary.collect_garbage()
    with unreal.ScopedSlowTask(6, "Building Refinery Digital Twin") as task:
        task.make_dialog(True)
        task.enter_progress_frame(1, "Importing 2K textures")
        import_textures()
        task.enter_progress_frame(1, "Building master material")
        master = build_master_material()
        task.enter_progress_frame(1, "Creating material instances")
        inst = build_instances(master)
        task.enter_progress_frame(1, "Importing Nanite meshes")
        meshes = import_meshes(inst)
        task.enter_progress_frame(1, "Laying out CDU-100")
        open_fresh_level()
        build_layout(meshes)
        frame_camera()
        task.enter_progress_frame(1, "Saving")
        LES.save_current_level()
        EAL.save_directory(G, only_if_is_dirty=True, recursive=True)
    log(f"DONE -> {LEVEL}")


main()
