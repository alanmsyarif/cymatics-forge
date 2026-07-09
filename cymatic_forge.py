bl_info = {
    "name": "Cymatic Forge",
    "author": "Amsy",
    "version": (1, 1, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Cymatics",
    "description": "Sound-reactive Chladni plate particle simulation using the Sample Sound Frequencies node",
    "category": "Object",
}

import bpy
import math
from bpy.props import (
    StringProperty, FloatProperty, IntProperty, BoolProperty,
    FloatVectorProperty, PointerProperty,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SAMPLE_SOUND_CANDIDATES = (
    "GeometryNodeSampleSoundFrequencies",
    "GeometryNodeSampleSound",
    "GeometryNodeSoundFrequencies",
)


def _new_sample_sound_node(nodes):
    """Create the 5.2 Sample Sound Frequencies node, whatever its idname is."""
    for tid in SAMPLE_SOUND_CANDIDATES:
        try:
            return nodes.new(tid)
        except RuntimeError:
            continue
    # Last resort: scan registered types for anything sound-sampling
    for cls in bpy.types.GeometryNode.__subclasses__():
        idname = getattr(cls, "bl_rna", None) and cls.bl_rna.identifier or ""
        if "SampleSound" in idname or ("Sound" in idname and "Freq" in idname):
            try:
                return nodes.new(idname)
            except RuntimeError:
                continue
    return None


def _find_socket(sockets, *keywords, exclude=()):
    """Fuzzy-find a socket whose name contains all keywords (case-insensitive)."""
    for s in sockets:
        n = s.name.lower()
        if all(k in n for k in keywords) and not any(x in n for x in exclude):
            return s
    return None


def _first_float_output(node):
    for s in node.outputs:
        if s.type in {'VALUE', 'FLOAT'}:
            return s
    return node.outputs[0] if node.outputs else None


def _setup_sample_node(node, sound, time_out, links, lo, hi):
    """Assign sound, wire time, set the frequency range on a sample node."""
    # Sound datablock: socket first, node property as fallback
    s_sock = _find_socket(node.inputs, "sound")
    assigned = False
    if s_sock is not None:
        try:
            s_sock.default_value = sound
            assigned = True
        except (AttributeError, TypeError):
            pass
    if not assigned and hasattr(node, "sound"):
        node.sound = sound
        assigned = True

    # Time in seconds
    t_sock = _find_socket(node.inputs, "time")
    if t_sock is not None and time_out is not None:
        links.new(time_out, t_sock)

    # Frequency range (Hz). Try start/end, low/high, min/max spellings.
    lo_sock = (_find_socket(node.inputs, "freq", "start")
               or _find_socket(node.inputs, "freq", "low")
               or _find_socket(node.inputs, "freq", "min")
               or _find_socket(node.inputs, "start")
               or _find_socket(node.inputs, "low"))
    hi_sock = (_find_socket(node.inputs, "freq", "end")
               or _find_socket(node.inputs, "freq", "high")
               or _find_socket(node.inputs, "freq", "max")
               or _find_socket(node.inputs, "end")
               or _find_socket(node.inputs, "high"))
    for sock, val in ((lo_sock, lo), (hi_sock, hi)):
        if sock is not None:
            try:
                sock.default_value = val
            except (AttributeError, TypeError):
                pass
    amp_out = (_find_socket(node.outputs, "amplitude")
               or _find_socket(node.outputs, "value")
               or _first_float_output(node))
    return assigned, amp_out


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------

class CymaticProps(bpy.types.PropertyGroup):
    sound_path: StringProperty(
        name="Audio File", subtype='FILE_PATH',
        description="Music track (.wav recommended, .mp3/.flac also work)")
    plate_size: FloatProperty(name="Plate Size", default=2.0, min=0.1,
                              description="Plate edge length in meters")
    resolution: IntProperty(name="Grid Resolution", default=200, min=10, max=600,
                            description="Points per side (200 = 40k particles)")
    point_radius: FloatProperty(name="Point Radius", default=0.004, min=0.0001,
                                precision=4)
    bass_lo: FloatProperty(name="Bass Low (Hz)", default=30.0, min=1.0)
    bass_hi: FloatProperty(name="Bass High (Hz)", default=150.0, min=1.0)
    mid_lo: FloatProperty(name="Mid Low (Hz)", default=150.0, min=1.0)
    mid_hi: FloatProperty(name="Mid High (Hz)", default=2000.0, min=1.0)
    full_lo: FloatProperty(name="Loudness Low (Hz)", default=20.0, min=1.0)
    full_hi: FloatProperty(name="Loudness High (Hz)", default=16000.0, min=1.0)
    amp_gain: FloatProperty(
        name="Amplitude Gain", default=0.25, min=0.001,
        description="Band amplitude that maps to full response; lower = more sensitive. Check spreadsheet values for your track")
    mode_min: IntProperty(name="Mode Min", default=1, min=1, max=16)
    mode_max: IntProperty(name="Mode Max", default=8, min=1, max=16)
    smooth_modes: BoolProperty(
        name="Smooth Mode Morphing", default=False,
        description="Skip integer rounding so patterns morph continuously instead of snapping")
    step: FloatProperty(name="Settle Speed", default=0.02, min=0.0,
                        description="Gradient-descent step toward nodal lines")
    jitter: FloatProperty(name="Jitter", default=0.005, min=0.0, precision=4,
                          description="Random boil driven by loudness")
    z_scale: FloatProperty(name="Z Displacement", default=0.03, min=0.0,
                           description="Physical plate bounce on Z")
    color_inner: FloatVectorProperty(
        name="Inner Color", subtype='COLOR', size=4,
        default=(0.55, 0.85, 1.0, 1.0), min=0.0, max=1.0,
        description="Emission color at the plate center")
    color_outer: FloatVectorProperty(
        name="Outer Color", subtype='COLOR', size=4,
        default=(0.35, 0.1, 0.9, 1.0), min=0.0, max=1.0,
        description="Emission color at the plate edge")
    dist_gamma: FloatProperty(
        name="Gradient Falloff", default=1.0, min=0.05, max=8.0,
        description="Power curve on the distance gradient; >1 pushes the inner color outward, <1 pulls the outer color inward")
    emission_strength: FloatProperty(name="Emission Gain", default=20.0, min=0.0)
    add_to_sequencer: BoolProperty(
        name="Add Audio to Sequencer + AV Sync", default=True)


# ------------------------------------------------------------------
# Build operator
# ------------------------------------------------------------------

class CYMATICS_OT_build(bpy.types.Operator):
    bl_idname = "cymatics.build"
    bl_label = "Build Cymatic Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.cymatic_forge
        if not p.sound_path:
            self.report({'ERROR'}, "Pick an audio file first")
            return {'CANCELLED'}
        try:
            sound = bpy.data.sounds.load(bpy.path.abspath(p.sound_path),
                                         check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Could not load audio: {e}")
            return {'CANCELLED'}

        # --- Object + modifier ---
        mesh = bpy.data.meshes.new("CymaticPlate")
        obj = bpy.data.objects.new("CymaticPlate", mesh)
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        tree = bpy.data.node_groups.new("CymaticSim", 'GeometryNodeTree')
        tree.is_modifier = True
        mod = obj.modifiers.new("CymaticSim", 'NODES')
        mod.node_group = tree

        ok, msg = self._build_tree(tree, sound, p)
        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}

        self._build_material(obj, tree, p)

        if p.add_to_sequencer:
            self._setup_audio_playback(context.scene, p)

        self.report({'INFO'}, f"Cymatic sim built ({msg})")
        return {'FINISHED'}

    # --------------------------------------------------------------
    def _build_tree(self, tree, sound, p):
        nodes, links = tree.nodes, tree.links
        iface = tree.interface
        iface.new_socket("Geometry", in_out='INPUT',
                         socket_type='NodeSocketGeometry')
        iface.new_socket("Geometry", in_out='OUTPUT',
                         socket_type='NodeSocketGeometry')

        L = p.plate_size / 2.0
        k = math.pi / L

        def nn(tid, x, y, label=""):
            n = nodes.new(tid)
            n.location = (x, y)
            if label:
                n.label = label
            return n

        def fmath(op, x, y, a=None, b=None, av=None, bv=None, label=""):
            n = nn("ShaderNodeMath", x, y, label)
            n.operation = op
            if a is not None:
                links.new(a, n.inputs[0])
            elif av is not None:
                n.inputs[0].default_value = av
            if b is not None:
                links.new(b, n.inputs[1])
            elif bv is not None:
                n.inputs[1].default_value = bv
            return n

        # --- Source points ---
        grid = nn("GeometryNodeMeshGrid", -1600, 400, "Plate")
        grid.inputs["Size X"].default_value = p.plate_size
        grid.inputs["Size Y"].default_value = p.plate_size
        grid.inputs["Vertices X"].default_value = p.resolution
        grid.inputs["Vertices Y"].default_value = p.resolution

        m2p = nn("GeometryNodeMeshToPoints", -1400, 400, "Particles")
        m2p.inputs["Radius"].default_value = p.point_radius
        links.new(grid.outputs["Mesh"], m2p.inputs["Mesh"])

        # --- Simulation zone ---
        sim_in = nn("GeometryNodeSimulationInput", -1150, 400)
        sim_out = nn("GeometryNodeSimulationOutput", 1400, 400)
        sim_in.pair_with_output(sim_out)
        links.new(m2p.outputs["Points"], sim_in.inputs["Geometry"])

        # --- Sound sampling ---
        stime = nn("GeometryNodeInputSceneTime", -1600, -100, "Scene Time")
        seconds = _find_socket(stime.outputs, "second") or stime.outputs[0]
        frame = _find_socket(stime.outputs, "frame")

        bands = {}
        node_id_used = None
        for i, (key, lo, hi) in enumerate((
                ("bass", p.bass_lo, p.bass_hi),
                ("mid", p.mid_lo, p.mid_hi),
                ("full", p.full_lo, p.full_hi))):
            sn = _new_sample_sound_node(nodes)
            if sn is None:
                return False, ("Sample Sound Frequencies node not found - "
                               "this requires Blender 5.2 LTS or newer")
            node_id_used = sn.bl_idname
            sn.location = (-1350, -50 - i * 240)
            sn.label = f"Sample {key} band"
            assigned, amp = _setup_sample_node(sn, sound, seconds, links, lo, hi)
            if amp is None:
                return False, f"No amplitude output found on {sn.bl_idname}"
            bands[key] = amp

        # --- Band -> mode numbers m, n and loudness ---
        def band_to_mode(amp, y):
            mr = nn("ShaderNodeMapRange", -1050, y)
            mr.inputs["From Min"].default_value = 0.0
            mr.inputs["From Max"].default_value = p.amp_gain
            mr.inputs["To Min"].default_value = p.mode_min
            mr.inputs["To Max"].default_value = p.mode_max
            mr.clamp = True
            links.new(amp, mr.inputs["Value"])
            if p.smooth_modes:
                return mr.outputs["Result"]
            r = fmath('ROUND', -880, y, a=mr.outputs["Result"], label="Snap")
            return r.outputs[0]

        m_out = band_to_mode(bands["bass"], -50)
        n_out = band_to_mode(bands["mid"], -290)

        loud_mr = nn("ShaderNodeMapRange", -1050, -530, "Loudness 0-1")
        loud_mr.inputs["From Min"].default_value = 0.0
        loud_mr.inputs["From Max"].default_value = p.amp_gain
        loud_mr.inputs["To Min"].default_value = 0.0
        loud_mr.inputs["To Max"].default_value = 1.0
        loud_mr.clamp = True
        links.new(bands["full"], loud_mr.inputs["Value"])
        loud = loud_mr.outputs["Result"]

        # --- Chladni field F and analytic gradient ---
        pos = nn("GeometryNodeInputPosition", -1050, 250)
        sep = nn("ShaderNodeSeparateXYZ", -880, 250)
        links.new(pos.outputs["Position"], sep.inputs["Vector"])
        X, Y = sep.outputs["X"], sep.outputs["Y"]

        nk = fmath('MULTIPLY', -700, -50, a=n_out, bv=k, label="n*k").outputs[0]
        mk = fmath('MULTIPLY', -700, -290, a=m_out, bv=k, label="m*k").outputs[0]

        a = fmath('MULTIPLY', -520, 320, a=X, b=nk, label="a=nkx").outputs[0]
        b = fmath('MULTIPLY', -520, 160, a=Y, b=mk, label="b=mky").outputs[0]
        c = fmath('MULTIPLY', -520, 0, a=X, b=mk, label="c=mkx").outputs[0]
        d = fmath('MULTIPLY', -520, -160, a=Y, b=nk, label="d=nky").outputs[0]

        cos_a = fmath('COSINE', -340, 360, a=a).outputs[0]
        sin_a = fmath('SINE', -340, 280, a=a).outputs[0]
        cos_b = fmath('COSINE', -340, 200, a=b).outputs[0]
        sin_b = fmath('SINE', -340, 120, a=b).outputs[0]
        cos_c = fmath('COSINE', -340, 40, a=c).outputs[0]
        sin_c = fmath('SINE', -340, -40, a=c).outputs[0]
        cos_d = fmath('COSINE', -340, -120, a=d).outputs[0]
        sin_d = fmath('SINE', -340, -200, a=d).outputs[0]

        t1 = fmath('MULTIPLY', -160, 300, a=cos_a, b=cos_b).outputs[0]
        t2 = fmath('MULTIPLY', -160, 160, a=cos_c, b=cos_d).outputs[0]
        F = fmath('SUBTRACT', 0, 230, a=t1, b=t2, label="Chladni F").outputs[0]

        neg_nk = fmath('MULTIPLY', -160, -20, a=nk, bv=-1.0).outputs[0]
        neg_mk = fmath('MULTIPLY', -160, -120, a=mk, bv=-1.0).outputs[0]

        # dF/dx = -nk sin(a)cos(b) + mk sin(c)cos(d)
        gx1 = fmath('MULTIPLY', 0, 60, a=sin_a, b=cos_b).outputs[0]
        gx1 = fmath('MULTIPLY', 140, 60, a=gx1, b=neg_nk).outputs[0]
        gx2 = fmath('MULTIPLY', 0, -40, a=sin_c, b=cos_d).outputs[0]
        gx2 = fmath('MULTIPLY', 140, -40, a=gx2, b=mk).outputs[0]
        dFdx = fmath('ADD', 280, 10, a=gx1, b=gx2, label="dF/dx").outputs[0]

        # dF/dy = -mk cos(a)sin(b) + nk cos(c)sin(d)
        gy1 = fmath('MULTIPLY', 0, -160, a=cos_a, b=sin_b).outputs[0]
        gy1 = fmath('MULTIPLY', 140, -160, a=gy1, b=neg_mk).outputs[0]
        gy2 = fmath('MULTIPLY', 0, -260, a=cos_c, b=sin_d).outputs[0]
        gy2 = fmath('MULTIPLY', 140, -260, a=gy2, b=nk).outputs[0]
        dFdy = fmath('ADD', 280, -210, a=gy1, b=gy2, label="dF/dy").outputs[0]

        # --- Motion: gradient descent on F^2, scaled by loudness ---
        step_s = fmath('MULTIPLY', 280, -400, a=loud, bv=-p.step,
                       label="-step*loud").outputs[0]
        ox = fmath('MULTIPLY', 440, 120, a=F, b=dFdx).outputs[0]
        ox = fmath('MULTIPLY', 580, 120, a=ox, b=step_s).outputs[0]
        oy = fmath('MULTIPLY', 440, -80, a=F, b=dFdy).outputs[0]
        oy = fmath('MULTIPLY', 580, -80, a=oy, b=step_s).outputs[0]
        oz_s = fmath('MULTIPLY', 440, -260, a=loud, bv=p.z_scale).outputs[0]
        oz = fmath('MULTIPLY', 580, -260, a=F, b=oz_s, label="Z bounce").outputs[0]

        comb = nn("ShaderNodeCombineXYZ", 740, 0, "Descent")
        links.new(ox, comb.inputs["X"])
        links.new(oy, comb.inputs["Y"])
        links.new(oz, comb.inputs["Z"])

        # --- Jitter (boil on loud passages) ---
        rand = nn("FunctionNodeRandomValue", 580, -430, "Jitter")
        rand.data_type = 'FLOAT_VECTOR'
        rand.inputs["Min"].default_value = (-1.0, -1.0, -0.2)
        rand.inputs["Max"].default_value = (1.0, 1.0, 0.2)
        if frame is not None:
            seed_sock = _find_socket(rand.inputs, "seed")
            if seed_sock is not None:
                links.new(frame, seed_sock)

        jit_s = fmath('MULTIPLY', 580, -600, a=loud, bv=p.jitter).outputs[0]
        jit_scale = nn("ShaderNodeVectorMath", 740, -430)
        jit_scale.operation = 'SCALE'
        links.new(rand.outputs["Value"], jit_scale.inputs[0])
        links.new(jit_s, jit_scale.inputs["Scale"])

        offset = nn("ShaderNodeVectorMath", 900, -120, "Total offset")
        offset.operation = 'ADD'
        links.new(comb.outputs["Vector"], offset.inputs[0])
        links.new(jit_scale.outputs["Vector"], offset.inputs[1])

        setpos = nn("GeometryNodeSetPosition", 1060, 300)
        links.new(sim_in.outputs["Geometry"], setpos.inputs["Geometry"])
        links.new(offset.outputs["Vector"], setpos.inputs["Offset"])

        # --- Clamp to plate bounds ---
        pos2 = nn("GeometryNodeInputPosition", 900, 560)
        vmax = nn("ShaderNodeVectorMath", 1060, 560)
        vmax.operation = 'MAXIMUM'
        vmax.inputs[1].default_value = (-L, -L, -1.0)
        links.new(pos2.outputs["Position"], vmax.inputs[0])
        vmin = nn("ShaderNodeVectorMath", 1200, 560)
        vmin.operation = 'MINIMUM'
        vmin.inputs[1].default_value = (L, L, 1.0)
        links.new(vmax.outputs["Vector"], vmin.inputs[0])
        clamp = nn("GeometryNodeSetPosition", 1250, 300, "Clamp bounds")
        links.new(setpos.outputs["Geometry"], clamp.inputs["Geometry"])
        links.new(vmin.outputs["Vector"], clamp.inputs["Position"])

        links.new(clamp.outputs["Geometry"], sim_out.inputs["Geometry"])

        # --- After zone: store loudness for the shader, set material ---
        store = nn("GeometryNodeStoreNamedAttribute", 1600, 400, "amp -> shader")
        store.domain = 'POINT'
        store.data_type = 'FLOAT'
        store.inputs["Name"].default_value = "amp"
        links.new(sim_out.outputs["Geometry"], store.inputs["Geometry"])
        links.new(loud, store.inputs["Value"])

        # Normalized distance from plate center (XY only), 0 center -> 1 edge
        pos3 = nn("GeometryNodeInputPosition", 1420, 700)
        flat = nn("ShaderNodeVectorMath", 1560, 700, "Drop Z")
        flat.operation = 'MULTIPLY'
        flat.inputs[1].default_value = (1.0, 1.0, 0.0)
        links.new(pos3.outputs["Position"], flat.inputs[0])
        dlen = nn("ShaderNodeVectorMath", 1700, 700, "Center dist")
        dlen.operation = 'LENGTH'
        links.new(flat.outputs["Vector"], dlen.inputs[0])
        dnorm = nn("ShaderNodeMapRange", 1840, 700, "Normalize 0-1")
        dnorm.inputs["From Min"].default_value = 0.0
        dnorm.inputs["From Max"].default_value = L
        dnorm.clamp = True
        links.new(dlen.outputs["Value"], dnorm.inputs["Value"])

        store_d = nn("GeometryNodeStoreNamedAttribute", 1780, 400,
                     "dist -> shader")
        store_d.domain = 'POINT'
        store_d.data_type = 'FLOAT'
        store_d.inputs["Name"].default_value = "dist"
        links.new(store.outputs["Geometry"], store_d.inputs["Geometry"])
        links.new(dnorm.outputs["Result"], store_d.inputs["Value"])

        setmat = nn("GeometryNodeSetMaterial", 1960, 400)
        links.new(store_d.outputs["Geometry"], setmat.inputs["Geometry"])

        gout = nn("NodeGroupOutput", 2140, 400)
        links.new(setmat.outputs["Geometry"], gout.inputs["Geometry"])

        self._setmat_node_name = setmat.name
        self._tree_name = tree.name
        return True, f"using {node_id_used}"

    # --------------------------------------------------------------
    def _build_material(self, obj, tree, p):
        mat = bpy.data.materials.new("CymaticGlow")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        out.location = (600, 0)
        emis = nt.nodes.new("ShaderNodeEmission")
        emis.location = (400, 0)

        # Color: distance from center -> falloff curve -> gradient ramp
        attr_d = nt.nodes.new("ShaderNodeAttribute")
        attr_d.location = (-400, 150)
        attr_d.attribute_name = "dist"
        gamma = nt.nodes.new("ShaderNodeMath")
        gamma.location = (-200, 150)
        gamma.operation = 'POWER'
        gamma.inputs[1].default_value = p.dist_gamma
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.location = (0, 150)
        ramp.color_ramp.elements[0].color = p.color_inner
        ramp.color_ramp.elements[1].color = p.color_outer
        nt.links.new(attr_d.outputs["Fac"], gamma.inputs[0])
        nt.links.new(gamma.outputs[0], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])

        # Strength: loudness attribute * gain
        attr_a = nt.nodes.new("ShaderNodeAttribute")
        attr_a.location = (-400, -150)
        attr_a.attribute_name = "amp"
        gain = nt.nodes.new("ShaderNodeMath")
        gain.location = (-200, -150)
        gain.operation = 'MULTIPLY'
        gain.inputs[1].default_value = p.emission_strength
        nt.links.new(attr_a.outputs["Fac"], gain.inputs[0])
        nt.links.new(gain.outputs[0], emis.inputs["Strength"])
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])

        setmat = tree.nodes.get(self._setmat_node_name)
        if setmat is not None:
            setmat.inputs["Material"].default_value = mat

    # --------------------------------------------------------------
    def _setup_audio_playback(self, scene, p):
        try:
            se = scene.sequence_editor or scene.sequence_editor_create()
            coll = getattr(se, "strips", None) or getattr(se, "sequences", None)
            strip = coll.new_sound("CymaticAudio",
                                   bpy.path.abspath(p.sound_path), 1, 1)
            dur = getattr(strip, "frame_final_duration", 0)
            if dur:
                scene.frame_end = max(scene.frame_end, int(dur))
        except Exception:
            pass
        try:
            scene.sync_mode = 'AUDIO_SYNC'
        except Exception:
            pass
        scene.frame_set(1)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

class CYMATICS_PT_panel(bpy.types.Panel):
    bl_label = "Cymatic Forge"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cymatics"

    def draw(self, context):
        p = context.scene.cymatic_forge
        col = self.layout.column(align=False)
        col.prop(p, "sound_path")

        box = col.box()
        box.label(text="Plate", icon='MESH_GRID')
        box.prop(p, "plate_size")
        box.prop(p, "resolution")
        box.prop(p, "point_radius")

        box = col.box()
        box.label(text="Frequency Bands", icon='RNDCURVE')
        row = box.row(align=True)
        row.prop(p, "bass_lo")
        row.prop(p, "bass_hi")
        row = box.row(align=True)
        row.prop(p, "mid_lo")
        row.prop(p, "mid_hi")
        row = box.row(align=True)
        row.prop(p, "full_lo")
        row.prop(p, "full_hi")
        box.prop(p, "amp_gain")

        box = col.box()
        box.label(text="Pattern", icon='FORCE_HARMONIC')
        row = box.row(align=True)
        row.prop(p, "mode_min")
        row.prop(p, "mode_max")
        box.prop(p, "smooth_modes")
        box.prop(p, "step")
        box.prop(p, "jitter")
        box.prop(p, "z_scale")

        box = col.box()
        box.label(text="Look", icon='SHADING_RENDERED')
        row = box.row(align=True)
        row.prop(p, "color_inner")
        row.prop(p, "color_outer")
        box.prop(p, "dist_gamma")
        box.prop(p, "emission_strength")

        col.prop(p, "add_to_sequencer")
        col.separator()
        col.operator("cymatics.build", icon='PLAY_SOUND')
        col.label(text="Play forward for live reaction;", icon='INFO')
        col.label(text="bake sim cache to scrub backward.")


classes = (CymaticProps, CYMATICS_OT_build, CYMATICS_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.cymatic_forge = PointerProperty(type=CymaticProps)


def unregister():
    del bpy.types.Scene.cymatic_forge
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
