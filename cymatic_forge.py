bl_info = {
    "name": "Cymatic Forge",
    "author": "Amsy",
    "version": (2, 0, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Cymatics",
    "description": "Photoreal, sound-driven Chladni plate: real sand grains with "
                   "weight, ballistic hops, piling and spill, studio-lit for Cycles",
    "category": "Object",
}

import bpy
import math
import random
from bpy.props import (
    StringProperty, FloatProperty, IntProperty, BoolProperty,
    EnumProperty, PointerProperty,
)

# ======================================================================
#  Physical model (summary)
#
#  The plate vibrates in a standing-wave mode F(x, y). Its local peak
#  acceleration, in units of g, is   Gamma = Drive * |F| / 2.
#  Sand only leaves the plate where Gamma > 1, so grains on the nodal lines
#  (F = 0) never move and grains on the antinodes are thrown up. Each grain is
#  a real particle with mass: it is launched, flies a ballistic arc under
#  gravity, tumbles, lands, bounces with low restitution and comes to rest.
#  Resting sand forms a volume-conserving height field (piles that grow
#  along the nodal lines); grains rest on that surface and avalanche when
#  the local slope is steeper than the angle of repose. Sand thrown past the
#  plate edges falls to the table and stays there.
#
#  Audio sets the drive (loudness envelope with plate ring-down, auto gain
#  and bass punch) and the vibration mode (spectral centroid -> resonance
#  table, with hysteresis so patterns hold instead of flickering).
# ======================================================================

GRAVITY = 9.81

# Resonance table: (m, n, sign) for
#   F = cos(n k x) cos(m k y) + sign * cos(m k x) cos(n k y),  k = pi / L
# Ordered from simplest (low pitch) to most intricate (high pitch).
MODES = (
    (1, 2, -1), (0, 2, 1), (1, 3, 1), (2, 3, -1),
    (1, 4, -1), (2, 4, 1), (1, 5, 1), (2, 5, -1),
    (3, 4, -1), (3, 5, 1), (1, 6, -1), (2, 6, 1),
    (2, 7, -1), (3, 7, 1), (4, 5, -1), (5, 7, 1),
)

PREFIX = "CF_"


# ======================================================================
#  Tiny node-building DSL
# ======================================================================

def _is_socket(x):
    return isinstance(x, bpy.types.NodeSocket)


class NB:
    """Wraps a node tree; every helper returns an output socket."""

    def __init__(self, tree):
        self.t = tree
        self.N = tree.nodes
        self.L = tree.links

    # -- core ---------------------------------------------------------
    def node(self, tid, label="", **props):
        n = self.N.new(tid)
        if label:
            n.label = label
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def feed(self, sock, val):
        if val is None:
            return
        if _is_socket(val):
            self.L.new(val, sock)
        else:
            sock.default_value = val

    @staticmethod
    def ins(node):
        return [s for s in node.inputs if s.enabled and not s.hide]

    @staticmethod
    def outs(node):
        return [s for s in node.outputs if s.enabled]

    # -- math ---------------------------------------------------------
    def m(self, op, a, b=None, c=None, label="", clamp=False):
        n = self.node("ShaderNodeMath", label, operation=op, use_clamp=clamp)
        for s, v in zip(n.inputs, (a, b, c)):
            self.feed(s, v)
        return n.outputs[0]

    def add(self, a, b):
        return self.m('ADD', a, b)

    def sub(self, a, b):
        return self.m('SUBTRACT', a, b)

    def mul(self, a, b):
        return self.m('MULTIPLY', a, b)

    def div(self, a, b):
        return self.m('DIVIDE', a, b)

    def mn(self, a, b):
        return self.m('MINIMUM', a, b)

    def mx(self, a, b):
        return self.m('MAXIMUM', a, b)

    def madd(self, a, b, c):
        return self.m('MULTIPLY_ADD', a, b, c)

    def clamp01(self, a):
        return self.m('ADD', a, 0.0, clamp=True)

    def v(self, op, a, b=None, c=None, scale=None, label=""):
        n = self.node("ShaderNodeVectorMath", label, operation=op)
        ins = self.ins(n)
        vec_ins = [s for s in ins if s.type == 'VECTOR']
        for s, val in zip(vec_ins, (a, b, c)):
            self.feed(s, val)
        if scale is not None:
            self.feed(n.inputs["Scale"], scale)
        if op in {'LENGTH', 'DOT_PRODUCT', 'DISTANCE'}:
            return n.outputs["Value"]
        return n.outputs["Vector"]

    def imath(self, op, a, b=None):
        n = self.node("FunctionNodeIntegerMath", operation=op)
        for s, val in zip(self.ins(n), (a, b)):
            self.feed(s, val)
        return n.outputs[0]

    def cmp(self, op, a, b, dtype='FLOAT'):
        n = self.node("FunctionNodeCompare", data_type=dtype, operation=op)
        ins = self.ins(n)
        self.feed(ins[0], a)
        self.feed(ins[1], b)
        return n.outputs[0]

    def lt(self, a, b):
        return self.cmp('LESS_THAN', a, b)

    def gt(self, a, b):
        return self.cmp('GREATER_THAN', a, b)

    def bmath(self, op, a, b=None):
        n = self.node("FunctionNodeBooleanMath", operation=op)
        ins = self.ins(n)
        self.feed(ins[0], a)
        if b is not None and len(ins) > 1:
            self.feed(ins[1], b)
        return n.outputs[0]

    def AND(self, a, b):
        return self.bmath('AND', a, b)

    def OR(self, a, b):
        return self.bmath('OR', a, b)

    def NOT(self, a):
        return self.bmath('NOT', a)

    def switch(self, typ, cond, f, t):
        n = self.node("GeometryNodeSwitch", input_type=typ)
        self.feed(n.inputs["Switch"], cond)
        self.feed(n.inputs["False"], f)
        self.feed(n.inputs["True"], t)
        return n.outputs[0]

    def xyz(self, x=0.0, y=0.0, z=0.0):
        n = self.node("ShaderNodeCombineXYZ")
        self.feed(n.inputs[0], x)
        self.feed(n.inputs[1], y)
        self.feed(n.inputs[2], z)
        return n.outputs[0]

    def sep(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self.feed(n.inputs[0], vec)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def rand(self, typ, lo, hi, id_=None, seed=None):
        n = self.node("FunctionNodeRandomValue", data_type=typ)
        ins = self.ins(n)
        names = {s.name: s for s in ins}
        self.feed(names["Min"], lo)
        self.feed(names["Max"], hi)
        if id_ is not None:
            self.feed(names["ID"], id_)
        if seed is not None:
            self.feed(names["Seed"], seed)
        return self.outs(n)[0]

    # -- geometry -----------------------------------------------------
    def pos(self):
        return self.node("GeometryNodeInputPosition").outputs[0]

    def index(self):
        return self.node("GeometryNodeInputIndex").outputs[0]

    def named(self, name, typ='FLOAT'):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=typ)
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def store(self, geo, name, val, typ='FLOAT', domain='POINT', sel=None):
        n = self.node("GeometryNodeStoreNamedAttribute", data_type=typ,
                      domain=domain)
        self.feed(n.inputs["Geometry"], geo)
        self.feed(n.inputs["Name"], name)
        self.feed(n.inputs["Value"], val)
        if sel is not None:
            self.feed(n.inputs["Selection"], sel)
        return n.outputs["Geometry"]

    def set_pos(self, geo, position=None, offset=None, sel=None):
        n = self.node("GeometryNodeSetPosition")
        self.feed(n.inputs["Geometry"], geo)
        self.feed(n.inputs["Position"], position)
        self.feed(n.inputs["Offset"], offset)
        if sel is not None:
            self.feed(n.inputs["Selection"], sel)
        return n.outputs["Geometry"]

    def sample_index(self, geo, val, idx, typ='FLOAT', domain='POINT'):
        n = self.node("GeometryNodeSampleIndex", data_type=typ, domain=domain,
                      clamp=True)
        self.feed(n.inputs["Geometry"], geo)
        self.feed(n.inputs["Value"], val)
        self.feed(n.inputs["Index"], idx)
        return n.outputs[0]

    def group(self, tree, **inputs):
        n = self.node("GeometryNodeGroup")
        n.node_tree = tree
        for k, val in inputs.items():
            self.feed(n.inputs[k], val)
        return n

    def sound(self, snd_sock, time, lo, hi, label=""):
        n = self.node("GeometryNodeSampleSoundFrequencies", label)
        self.feed(n.inputs["Sound"], snd_sock)
        self.feed(n.inputs["Time"], time)
        self.feed(n.inputs["Low"], lo)
        self.feed(n.inputs["High"], hi)
        try:
            n.inputs["FFT Size"].default_value = '4096'
        except (KeyError, TypeError):
            pass
        return n.outputs["Amplitude"]


def _layout(tree, dx=230, dy=170):
    """Topological column layout so generated trees stay readable."""
    nodes = list(tree.nodes)
    depth = {n: 0 for n in nodes}
    incoming = {n: [] for n in nodes}
    for lk in tree.links:
        incoming[lk.to_node].append(lk.from_node)
    for _ in range(len(nodes)):
        changed = False
        for n in nodes:
            for src in incoming[n]:
                if depth[n] < depth[src] + 1:
                    depth[n] = depth[src] + 1
                    changed = True
        if not changed:
            break
    cols = {}
    for n in nodes:
        cols.setdefault(depth[n], []).append(n)
    for d, col in cols.items():
        for i, n in enumerate(col):
            n.location = (d * dx, -i * dy + len(col) * dy / 2)


def _new_group(name, inputs=(), outputs=()):
    """inputs/outputs: (name, socket_type, default, min, max)."""
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    for spec in inputs:
        nm, st = spec[0], spec[1]
        it = tree.interface.new_socket(nm, in_out='INPUT', socket_type=st)
        if len(spec) > 2 and spec[2] is not None and hasattr(it, "default_value"):
            it.default_value = spec[2]
        if len(spec) > 3 and hasattr(it, "min_value"):
            it.min_value = spec[3]
        if len(spec) > 4 and hasattr(it, "max_value"):
            it.max_value = spec[4]
    for nm, st in outputs:
        tree.interface.new_socket(nm, in_out='OUTPUT', socket_type=st)
    b = NB(tree)
    gi = b.node("NodeGroupInput")
    go = b.node("NodeGroupOutput")
    return tree, b, gi, go


# ======================================================================
#  Node groups
# ======================================================================

def build_chladni_group():
    """Mode shape F and its analytic gradient at a position."""
    tree, b, gi, go = _new_group(
        PREFIX + "Chladni Mode",
        inputs=(("Position", 'NodeSocketVector', None),
                ("M", 'NodeSocketFloat', 1.0),
                ("N", 'NodeSocketFloat', 2.0),
                ("Sign", 'NodeSocketFloat', -1.0),
                ("K", 'NodeSocketFloat', 10.0)),
        outputs=(("F", 'NodeSocketFloat'), ("Gradient", 'NodeSocketVector')))
    X, Y, _ = b.sep(gi.outputs["Position"])
    mk = b.mul(gi.outputs["M"], gi.outputs["K"])
    nk = b.mul(gi.outputs["N"], gi.outputs["K"])
    S = gi.outputs["Sign"]
    a = b.mul(X, nk)   # n k x
    bb = b.mul(Y, mk)  # m k y
    c = b.mul(X, mk)   # m k x
    d = b.mul(Y, nk)   # n k y
    ca, sa = b.m('COSINE', a), b.m('SINE', a)
    cb, sb = b.m('COSINE', bb), b.m('SINE', bb)
    cc, sc = b.m('COSINE', c), b.m('SINE', c)
    cd, sd = b.m('COSINE', d), b.m('SINE', d)
    F = b.madd(b.mul(cc, cd), S, b.mul(ca, cb))
    # dF/dx = -nk sa cb - S mk sc cd
    gx = b.sub(b.mul(b.mul(sa, cb), b.mul(nk, -1.0)),
               b.mul(b.mul(sc, cd), b.mul(mk, S)))
    # dF/dy = -mk ca sb - S nk cc sd
    gy = b.sub(b.mul(b.mul(ca, sb), b.mul(mk, -1.0)),
               b.mul(b.mul(cc, sd), b.mul(nk, S)))
    b.L.new(F, go.inputs["F"])
    b.L.new(b.xyz(gx, gy, 0.0), go.inputs["Gradient"])
    _layout(tree)
    return tree


def build_heightfield_group():
    """Volume-conserving sand height field from resting grains.

    Each resting grain deposits its solid volume (divided by the packing
    fraction) into the nearest cell of a grid mesh (with a stable per-grain
    sub-cell jitter so the deposit is smooth), the result is blurred and
    becomes the height of the pile. Output mesh has 'cf_h' and z = h.
    """
    tree, b, gi, go = _new_group(
        PREFIX + "Sand Height Field",
        inputs=(("Grains", 'NodeSocketGeometry', None),
                ("Half Size", 'NodeSocketFloat', 0.15),
                ("Resolution", 'NodeSocketInt', 160, 16, 1024),
                ("Packing", 'NodeSocketFloat', 0.58),
                ("Smoothing", 'NodeSocketInt', 2, 0, 40)),
        outputs=(("Height Field", 'NodeSocketGeometry'),))
    Lh = gi.outputs["Half Size"]
    G = gi.outputs["Resolution"]
    size = b.mul(Lh, 2.0)
    cs = b.div(size, b.sub(G, 1.0))           # cell spacing

    grid = b.node("GeometryNodeMeshGrid")
    b.feed(grid.inputs["Size X"], size)
    b.feed(grid.inputs["Size Y"], size)
    b.feed(grid.inputs["Vertices X"], G)
    b.feed(grid.inputs["Vertices Y"], G)

    # Cell points: weight 0, own index as cell id.
    cellpts = b.node("GeometryNodeMeshToPoints")
    b.feed(cellpts.inputs["Mesh"], grid.outputs["Mesh"])
    cg = b.store(cellpts.outputs["Points"], "cf_cell", b.index(), 'INT')
    cg = b.store(cg, "cf_w", 0.0)
    cg = b.store(cg, "cf_iscell", True, 'BOOLEAN')

    # Grains: weight = bulk volume / cell area, cell id from jittered pos.
    grains = gi.outputs["Grains"]
    gp = b.pos()
    gx, gy, _ = b.sep(gp)
    idn = b.node("GeometryNodeInputID").outputs[0]
    jx = b.rand('FLOAT', -0.5, 0.5, idn, 7)
    jy = b.rand('FLOAT', -0.5, 0.5, idn, 11)
    u = b.add(b.div(b.add(gx, Lh), cs), jx)
    v = b.add(b.div(b.add(gy, Lh), cs), jy)
    gm1 = b.sub(G, 1.0)
    iu = b.m('ROUND', b.mx(b.mn(u, gm1), 0.0))
    iv = b.m('ROUND', b.mx(b.mn(v, gm1), 0.0))
    cell = b.madd(iu, G, iv)
    r = b.named("cf_rad")
    vol = b.mul(b.m('POWER', r, 3.0), 4.18879 * 0.82)  # 4/3 pi r^3, flattened
    w = b.div(vol, b.mul(b.mul(cs, cs), gi.outputs["Packing"]))
    gg = b.store(grains, "cf_cell", b.m('ROUND', cell), 'INT')
    gg = b.store(gg, "cf_w", w)

    join = b.node("GeometryNodeJoinGeometry")
    b.L.new(gg, join.inputs[0])
    b.L.new(cg, join.inputs[0])

    acc = b.node("GeometryNodeAccumulateField", data_type='FLOAT',
                 domain='POINT')
    b.feed(acc.inputs["Value"], b.named("cf_w"))
    b.feed(acc.inputs["Group ID"], b.named("cf_cell", 'INT'))
    joined = b.store(join.outputs[0], "cf_hraw", acc.outputs["Total"])

    sepg = b.node("GeometryNodeSeparateGeometry", domain='POINT')
    b.feed(sepg.inputs["Geometry"], joined)
    b.feed(sepg.inputs["Selection"], b.named("cf_iscell", 'BOOLEAN'))
    cells_only = sepg.outputs["Selection"]

    # Back onto the grid mesh (same order), blur, displace.
    hraw = b.sample_index(cells_only, b.named("cf_hraw"), b.index())
    mesh = b.store(grid.outputs["Mesh"], "cf_h", hraw)
    blur = b.node("GeometryNodeBlurAttribute", data_type='FLOAT')
    b.feed(blur.inputs["Value"], b.named("cf_h"))
    b.feed(blur.inputs["Iterations"], gi.outputs["Smoothing"])
    mesh = b.store(mesh, "cf_h", blur.outputs[0])
    px, py, _ = b.sep(b.pos())
    mesh = b.set_pos(mesh, position=b.xyz(px, py, b.named("cf_h")))
    b.L.new(mesh, go.inputs[0])
    _layout(tree)
    return tree


def build_sample_height_group():
    """Bilinear height + slope of the height field at a position (field)."""
    tree, b, gi, go = _new_group(
        PREFIX + "Sample Sand Height",
        inputs=(("Height Field", 'NodeSocketGeometry', None),
                ("Position", 'NodeSocketVector', None),
                ("Half Size", 'NodeSocketFloat', 0.15),
                ("Resolution", 'NodeSocketInt', 160, 16, 1024)),
        outputs=(("Height", 'NodeSocketFloat'),
                 ("Slope", 'NodeSocketVector')))
    Lh = gi.outputs["Half Size"]
    G = gi.outputs["Resolution"]
    hf = gi.outputs["Height Field"]
    cs = b.div(b.mul(Lh, 2.0), b.sub(G, 1.0))
    x, y, _ = b.sep(gi.outputs["Position"])
    gm2 = b.sub(G, 1.0001)
    u = b.mx(b.mn(b.div(b.add(x, Lh), cs), gm2), 0.0)
    v = b.mx(b.mn(b.div(b.add(y, Lh), cs), gm2), 0.0)
    iu = b.m('FLOOR', u)
    iv = b.m('FLOOR', v)
    fu = b.sub(u, iu)
    fv = b.sub(v, iv)
    i00 = b.madd(iu, G, iv)
    h_attr = b.named("cf_h")

    def hs(off):
        return b.sample_index(hf, h_attr, b.m('ROUND', b.add(i00, off)))
    h00 = hs(0.0)
    h01 = hs(1.0)
    h10 = b.sample_index(hf, h_attr, b.m('ROUND', b.add(i00, G)))
    h11 = b.sample_index(hf, h_attr, b.m('ROUND', b.add(b.add(i00, G), 1.0)))

    def lerp(p, q, t):
        return b.madd(b.sub(q, p), t, p)
    h0 = lerp(h00, h01, fv)
    h1 = lerp(h10, h11, fv)
    h = lerp(h0, h1, fu)
    dhdx = b.div(b.sub(h1, h0), cs)
    dhdy = b.div(lerp(b.sub(h01, h00), b.sub(h11, h10), fu), cs)
    b.L.new(h, go.inputs["Height"])
    b.L.new(b.xyz(dhdx, dhdy, 0.0), go.inputs["Slope"])
    _layout(tree)
    return tree


def build_grain_step_group(chladni, sampler):
    """One physics sub-step for every grain (ballistics, launch, landing,
    bounce, rest, avalanche, spill)."""
    tree, b, gi, go = _new_group(
        PREFIX + "Grain Step",
        inputs=(("Grains", 'NodeSocketGeometry', None),
                ("Height Field", 'NodeSocketGeometry', None),
                ("dt", 'NodeSocketFloat', 0.01),
                ("Seed", 'NodeSocketInt', 0),
                ("Drive", 'NodeSocketFloat', 0.0),
                ("M", 'NodeSocketFloat', 1.0),
                ("N", 'NodeSocketFloat', 2.0),
                ("Sign", 'NodeSocketFloat', -1.0),
                ("Half Size", 'NodeSocketFloat', 0.15),
                ("Resolution", 'NodeSocketInt', 160),
                ("Table Z", 'NodeSocketFloat', -0.14),
                ("Gravity", 'NodeSocketFloat', GRAVITY),
                ("Hop Height", 'NodeSocketFloat', 0.004),
                ("Launch Rate", 'NodeSocketFloat', 40.0),
                ("Scatter", 'NodeSocketFloat', 0.45),
                ("Node Attraction", 'NodeSocketFloat', 0.35),
                ("Restitution", 'NodeSocketFloat', 0.28),
                ("Friction", 'NodeSocketFloat', 0.55),
                ("Repose Tan", 'NodeSocketFloat', 0.65),
                ("Avalanche Speed", 'NodeSocketFloat', 0.03),
                ("Creep", 'NodeSocketFloat', 0.01)),
        outputs=(("Grains", 'NodeSocketGeometry'),))
    I = gi.outputs
    dt = I["dt"]
    g = I["Gravity"]
    Lh = I["Half Size"]

    P = b.pos()
    px, py, pz = b.sep(P)
    vel = b.named("cf_vel", 'FLOAT_VECTOR')
    vx, vy, vz = b.sep(vel)
    r = b.named("cf_rad")
    nest = b.named("cf_nest")            # 0..1 how deep the grain nests
    off = b.named("cf_off", 'BOOLEAN')   # fell off the plate for good
    idn = b.node("GeometryNodeInputID").outputs[0]

    inside = b.AND(b.lt(b.m('ABSOLUTE', px), Lh), b.lt(b.m('ABSOLUTE', py), Lh))
    onplate = b.AND(inside, b.NOT(off))

    smp = b.group(sampler, **{"Height Field": I["Height Field"], "Position": P,
                              "Half Size": Lh, "Resolution": I["Resolution"]})
    h = smp.outputs["Height"]
    slope = smp.outputs["Slope"]
    # Rest height of the grain centre: on the pile surface, nested a bit.
    pile_rest = b.add(b.mx(b.sub(h, b.mul(r, 0.55)), 0.0),
                      b.mul(r, b.sub(0.9, b.mul(nest, 0.35))))
    table_rest = b.add(I["Table Z"], b.mul(r, 0.75))
    floor_z = b.switch('FLOAT', onplate, table_rest, pile_rest)

    height_above = b.sub(pz, floor_z)
    grounded = b.AND(b.lt(height_above, b.mul(r, 0.25)),
                     b.lt(vz, 0.02))

    # --- plate excitation ---------------------------------------------
    # A real plate is never perfect: slight thickness/mounting variation bends
    # the nodal lines a little and breaks the textbook symmetry.
    wx = b.mul(Lh, b.add(b.mul(b.m('SINE', b.add(b.mul(b.div(py, Lh), 2.1), 0.7)), 0.012), 0.006))
    wy = b.mul(Lh, b.sub(b.mul(b.m('SINE', b.add(b.mul(b.div(px, Lh), 1.6), 1.9)), 0.010), 0.004))
    Pw = b.v('ADD', P, b.xyz(wx, wy, 0.0))
    ch = b.group(chladni, Position=Pw, M=I["M"], N=I["N"], Sign=I["Sign"],
                 K=b.div(math.pi, Lh))
    F = ch.outputs["F"]
    gradF = ch.outputs["Gradient"]
    gamma = b.mul(I["Drive"], b.mul(b.m('ABSOLUTE', F), 0.5))
    # per-grain static friction: a few angular grains wedge in micro-scratches
    # and need more than 1 g to break free -> natural stragglers around ridges
    grip = b.named("cf_grip")
    excess = b.mx(b.sub(gamma, b.madd(grip, 2.5, 1.0)), 0.0)
    u1 = b.rand('FLOAT', 0.0, 1.0, idn, I["Seed"])
    p_launch = b.clamp01(b.mul(b.mul(excess, I["Launch Rate"]), dt))
    launch = b.AND(b.AND(grounded, onplate), b.lt(u1, p_launch))

    # Launch velocity: vertical from hop height ~ excess, sideways scatter
    # plus a drift down the |F| gradient (towards the nodal line).
    u2 = b.rand('FLOAT', 0.25, 1.0, idn, b.imath('ADD', I["Seed"], 101))
    hop = b.mul(I["Hop Height"], b.mul(b.mn(excess, 2.0), u2))
    vlz = b.m('SQRT', b.mul(b.mul(2.0, g), hop))
    ang = b.rand('FLOAT', 0.0, 6.2831853, idn, b.imath('ADD', I["Seed"], 202))
    sc = b.mul(vlz, I["Scatter"])
    rx = b.mul(b.m('COSINE', ang), sc)
    ry = b.mul(b.m('SINE', ang), sc)
    gnorm = b.v('NORMALIZE', gradF)
    toward = b.v('SCALE', gnorm, scale=b.mul(b.m('SIGN', F), -1.0))
    tx, ty, _ = b.sep(toward)
    att = b.mul(vlz, I["Node Attraction"])
    vlx = b.madd(tx, att, rx)
    vly = b.madd(ty, att, ry)

    # --- integrate ----------------------------------------------------
    # state flags
    resting = b.AND(grounded, b.NOT(launch))
    flying = b.NOT(resting)
    v0x = b.switch('FLOAT', launch, vx, vlx)
    v0y = b.switch('FLOAT', launch, vy, vly)
    v0z = b.switch('FLOAT', launch, vz, vlz)
    # gravity + light air drag
    drag = b.sub(1.0, b.mul(dt, 0.4))
    v1x = b.mul(v0x, drag)
    v1y = b.mul(v0y, drag)
    v1z = b.sub(b.mul(v0z, drag), b.mul(g, dt))
    nx = b.madd(v1x, dt, px)
    ny = b.madd(v1y, dt, py)
    nz = b.madd(v1z, dt, pz)

    # landing / bounce against the floor height at the old position
    below = b.lt(nz, floor_z)
    e = I["Restitution"]
    vstop = b.m('SQRT', b.mul(b.mul(g, r), 4.0))
    bounce_vz = b.mul(b.m('ABSOLUTE', v1z), e)
    keep = b.sub(1.0, I["Friction"])
    alive = b.gt(bounce_vz, vstop)
    bvx = b.switch('FLOAT', alive, 0.0, b.mul(v1x, keep))
    bvy = b.switch('FLOAT', alive, 0.0, b.mul(v1y, keep))
    bvz = b.switch('FLOAT', alive, 0.0, bounce_vz)
    fvx = b.switch('FLOAT', below, v1x, bvx)
    fvy = b.switch('FLOAT', below, v1y, bvy)
    fvz = b.switch('FLOAT', below, v1z, bvz)
    fz = b.switch('FLOAT', below, nz, floor_z)

    # resting grains avalanche: probe the pile in a random direction and
    # slide that way if the drop is steeper than the angle of repose. (A
    # gradient test misses crests, where the measured slope is zero, which
    # lets sand stack into unphysical spikes.)
    cs = b.div(b.mul(Lh, 2.0), b.sub(I["Resolution"], 1.0))
    # random probe length breaks any locking to the height-field lattice
    probe_d = b.mul(b.mx(cs, b.mul(r, 2.0)),
                    b.rand('FLOAT', 0.6, 2.2, idn, b.imath('ADD', I["Seed"], 505)))
    pang = b.rand('FLOAT', 0.0, 6.2831853, idn, b.imath('ADD', I["Seed"], 404))
    pdx = b.m('COSINE', pang)
    pdy = b.m('SINE', pang)
    probe = b.v('ADD', P, b.xyz(b.mul(pdx, probe_d), b.mul(pdy, probe_d), 0.0))
    smp2 = b.group(sampler, **{"Height Field": I["Height Field"], "Position": probe,
                               "Half Size": Lh, "Resolution": I["Resolution"]})
    drop = b.div(b.sub(h, smp2.outputs["Height"]), probe_d)
    over = b.mn(b.mx(b.sub(drop, I["Repose Tan"]), 0.0), 1.0)
    slide = b.mul(b.mul(over, I["Avalanche Speed"]), dt)
    dnx = b.mul(pdx, slide)
    dny = b.mul(pdy, slide)
    # tiny settling shiver while the plate is excited but below 1 g
    shiver = b.mul(b.mul(b.clamp01(b.mul(gamma, 1.5)), dt), 0.0004)
    sj = b.rand('FLOAT_VECTOR', (-1.0, -1.0, 0.0), (1.0, 1.0, 0.0), idn,
                b.imath('ADD', I["Seed"], 303))
    sjx, sjy, _ = b.sep(sj)
    # vibration creep: below 1 g grains still slide/roll down the oscillating
    # plate tilt towards the nodal line (stalls where the plate is still)
    dist = b.div(b.m('ABSOLUTE', F), b.mx(b.v('LENGTH', gradF), 1e-6))
    # ...but only while the pile ahead is not already at the angle of repose:
    # sand packs up against the ridge and builds a true repose-angle mound.
    ahead = b.v('ADD', P, b.v('SCALE', toward, scale=probe_d))
    smp3 = b.group(sampler, **{"Height Field": I["Height Field"], "Position": ahead,
                               "Half Size": Lh, "Resolution": I["Resolution"]})
    rise = b.div(b.sub(smp3.outputs["Height"], h), probe_d)
    free = b.clamp01(b.div(b.sub(I["Repose Tan"], rise), b.mul(I["Repose Tan"], 0.25)))
    free = b.mul(free, b.mx(b.clamp01(b.sub(1.0, b.mul(grip, 1.6))), 0.1))
    creep = b.mn(b.mul(b.mul(b.mul(b.clamp01(gamma), free), I["Creep"]), dt),
                 b.mx(b.sub(dist, b.mul(r, 0.5)), 0.0))
    cx = b.mul(tx, creep)
    cy = b.mul(ty, creep)
    rx2 = b.add(b.madd(sjx, shiver, px),
                b.switch('FLOAT', onplate, 0.0, b.add(dnx, cx)))
    ry2 = b.add(b.madd(sjy, shiver, py),
                b.switch('FLOAT', onplate, 0.0, b.add(dny, cy)))

    outx = b.switch('FLOAT', flying, rx2, nx)
    outy = b.switch('FLOAT', flying, ry2, ny)
    outz = b.switch('FLOAT', flying, floor_z, fz)
    ovx = b.switch('FLOAT', flying, 0.0, fvx)
    ovy = b.switch('FLOAT', flying, 0.0, fvy)
    ovz = b.switch('FLOAT', flying, 0.0, fvz)

    geo = I["Grains"]
    rot = b.named("cf_rot", 'FLOAT_VECTOR')
    spin = b.named("cf_spin", 'FLOAT_VECTOR')
    cap = b.node("GeometryNodeCaptureAttribute", domain='POINT')
    for typ, nm in (('VECTOR', "Pos"), ('VECTOR', "Vel"), ('VECTOR', "Rot"), ('BOOLEAN', "Off")):
        cap.capture_items.new(typ, nm)
    b.feed(cap.inputs["Geometry"], geo)
    b.feed(cap.inputs["Pos"], b.xyz(outx, outy, outz))
    b.feed(cap.inputs["Vel"], b.xyz(ovx, ovy, ovz))
    b.feed(cap.inputs["Rot"], b.switch('VECTOR', flying, rot,
                                       b.v('ADD', rot, b.v('SCALE', spin, scale=dt))))
    b.feed(cap.inputs["Off"], b.OR(off, b.lt(pz, -0.0015)))
    geo = cap.outputs["Geometry"]
    geo = b.store(geo, "cf_vel", cap.outputs["Vel"], 'FLOAT_VECTOR')
    geo = b.store(geo, "cf_rot", cap.outputs["Rot"], 'FLOAT_VECTOR')
    geo = b.store(geo, "cf_off", cap.outputs["Off"], 'BOOLEAN')
    geo = b.set_pos(geo, position=cap.outputs["Pos"])
    b.L.new(geo, go.inputs[0])
    _layout(tree)
    return tree


def build_audio_group():
    """Audio analysis -> (level, bass, centroid) single values."""
    tree, b, gi, go = _new_group(
        PREFIX + "Audio Analysis",
        inputs=(("Sound", 'NodeSocketSound', None),
                ("Time", 'NodeSocketFloat', 0.0),
                ("Low Hz", 'NodeSocketFloat', 120.0),
                ("High Hz", 'NodeSocketFloat', 4000.0),
                ("Bass Low Hz", 'NodeSocketFloat', 30.0),
                ("Bass High Hz", 'NodeSocketFloat', 140.0)),
        outputs=(("Loudness", 'NodeSocketFloat'),
                 ("Bass", 'NodeSocketFloat'),
                 ("Pitch", 'NodeSocketFloat')))
    snd = gi.outputs["Sound"]
    t = gi.outputs["Time"]
    loud = b.sound(snd, t, 20.0, 16000.0, "Loudness")
    bass = b.sound(snd, t, gi.outputs["Bass Low Hz"], gi.outputs["Bass High Hz"],
                   "Bass")
    # K log-spaced bands; spectral centroid (power weighted) in 0..1
    K = 8
    lo = gi.outputs["Low Hz"]
    ratio = b.div(gi.outputs["High Hz"], lo)
    num = None
    den = None
    for i in range(K):
        f0 = b.mul(lo, b.m('POWER', ratio, i / K))
        f1 = b.mul(lo, b.m('POWER', ratio, (i + 1) / K))
        a = b.sound(snd, t, f0, f1, f"Band {i + 1}")
        pw = b.mul(a, a)
        num = b.mul(pw, i / (K - 1)) if num is None else b.madd(pw, i / (K - 1), num)
        den = pw if den is None else b.add(den, pw)
    pitch = b.div(num, b.mx(den, 1e-9))
    b.L.new(loud, go.inputs["Loudness"])
    b.L.new(bass, go.inputs["Bass"])
    b.L.new(pitch, go.inputs["Pitch"])
    _layout(tree)
    return tree


def build_mode_table_group():
    """Mode index -> (M, N, Sign)."""
    tree, b, gi, go = _new_group(
        PREFIX + "Mode Table",
        inputs=(("Mode", 'NodeSocketInt', 1, 1, len(MODES)),),
        outputs=(("M", 'NodeSocketFloat'), ("N", 'NodeSocketFloat'),
                 ("Sign", 'NodeSocketFloat')))
    idx = b.imath('SUBTRACT', gi.outputs["Mode"], 1)
    for comp, name in enumerate(("M", "N", "Sign")):
        sw = b.node("GeometryNodeIndexSwitch", data_type='FLOAT')
        while len([s for s in sw.inputs if s.name.isdigit()]) < len(MODES):
            sw.index_switch_items.new()
        b.feed(sw.inputs["Index"], idx)
        for i, md in enumerate(MODES):
            sw.inputs[str(i)].default_value = float(md[comp])
        b.L.new(sw.outputs[0], go.inputs[name])
    _layout(tree)
    return tree


# ======================================================================
#  Main tree
# ======================================================================

MAIN_INPUTS = (
    # (panel, name, socket, default, min, max, description)
    ("Audio", "Sound", 'NodeSocketSound', None, None, None,
     "Music track driving the plate"),
    ("Audio", "Audio Offset", 'NodeSocketFloat', 0.0, -60.0, 60.0,
     "Seconds added to the playhead when sampling the track"),
    ("Audio", "Drive", 'NodeSocketFloat', 6.0, 0.0, 20.0,
     "Peak plate acceleration (in g) at full loudness. Sand only moves above 1 g"),
    ("Audio", "Bass Punch", 'NodeSocketFloat', 2.5, 0.0, 20.0,
     "Extra acceleration (g) from bass transients - kicks make sand jump"),
    ("Audio", "Ring Down", 'NodeSocketFloat', 0.25, 0.01, 5.0,
     "Seconds for the plate to stop ringing after the sound drops"),
    ("Audio", "Pitch Low Hz", 'NodeSocketFloat', 120.0, 1.0, 20000.0,
     "Bottom of the range that selects vibration modes"),
    ("Audio", "Pitch High Hz", 'NodeSocketFloat', 4000.0, 1.0, 20000.0,
     "Top of the range that selects vibration modes"),
    ("Audio", "Bass Low Hz", 'NodeSocketFloat', 30.0, 1.0, 20000.0, ""),
    ("Audio", "Bass High Hz", 'NodeSocketFloat', 140.0, 1.0, 20000.0, ""),
    ("Pattern", "Mode Min", 'NodeSocketInt', 2, 1, len(MODES),
     "Simplest resonance used (low pitch)"),
    ("Pattern", "Mode Max", 'NodeSocketInt', 12, 1, len(MODES),
     "Most intricate resonance used (high pitch)"),
    ("Pattern", "Mode Hysteresis", 'NodeSocketFloat', 0.8, 0.0, 4.0,
     "How far the pitch must move before the plate jumps to a new resonance"),
    ("Sand", "Grain Count", 'NodeSocketInt', 450000, 100, 5000000,
     "Number of grains sprinkled at the first frame"),
    ("Sand", "Grain Radius", 'NodeSocketFloat', 0.00022, 0.00002, 0.02,
     "Mean grain radius in metres (fine sand ~0.15-0.4 mm)"),
    ("Sand", "Size Variation", 'NodeSocketFloat', 0.3, 0.0, 0.9, ""),
    ("Sand", "Sprinkle Margin", 'NodeSocketFloat', 0.03, 0.0, 0.5,
     "Fraction of the plate edge left clean when sprinkling"),
    ("Sand", "Seed", 'NodeSocketInt', 0, 0, 100000, ""),
    ("Physics", "Gravity", 'NodeSocketFloat', GRAVITY, 0.0, 100.0, ""),
    ("Physics", "Hop Height", 'NodeSocketFloat', 0.004, 0.0, 0.2,
     "Typical hop height (m) of a grain thrown off an antinode at 2 g excess"),
    ("Physics", "Launch Rate", 'NodeSocketFloat', 40.0, 0.0, 500.0,
     "How eagerly grains leave the plate per second per g of excess"),
    ("Physics", "Scatter", 'NodeSocketFloat', 0.45, 0.0, 3.0,
     "Random sideways speed relative to the vertical launch speed"),
    ("Physics", "Node Attraction", 'NodeSocketFloat', 0.35, 0.0, 3.0,
     "Sideways speed towards the nearest nodal line"),
    ("Physics", "Creep", 'NodeSocketFloat', 0.012, 0.0, 1.0,
     "Sliding speed (m/s) of resting sand towards the nodes while the plate vibrates"),
    ("Physics", "Restitution", 'NodeSocketFloat', 0.28, 0.0, 0.95,
     "Bounciness of a grain landing on metal/sand"),
    ("Physics", "Friction", 'NodeSocketFloat', 0.55, 0.0, 1.0,
     "Fraction of sideways speed lost per bounce"),
    ("Physics", "Angle of Repose", 'NodeSocketFloat', 33.0, 5.0, 60.0,
     "Steepest stable pile slope in degrees (dry sand ~30-35)"),
    ("Physics", "Substeps", 'NodeSocketInt', 6, 1, 64,
     "Physics sub-steps per 1/24 s (scaled automatically for other frame rates)"),
    ("Physics", "Table Height", 'NodeSocketFloat', 0.14, 0.0, 10.0,
     "Drop from the plate top to the table where spilled sand lands"),
    ("Plate", "Plate Size", 'NodeSocketFloat', 0.30, 0.01, 100.0,
     "Plate edge length (m). Must match the plate object"),
    ("Plate", "Pile Resolution", 'NodeSocketInt', 256, 16, 1024,
     "Cells per side of the sand height field"),
    ("Look", "Grain Material", 'NodeSocketMaterial', None, None, None, ""),
    ("Look", "Pile Material", 'NodeSocketMaterial', None, None, None, ""),
    ("Look", "Viewport Full Grains", 'NodeSocketBool', False, None, None,
     "Show real grain meshes in the viewport (slow); otherwise points"),
)


def build_main_tree(name, grain_shapes_tree):
    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    tree.is_modifier = True
    iface = tree.interface
    iface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    iface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    panels = {}
    for pnl, nm, st, dv, lo, hi, desc in MAIN_INPUTS:
        if pnl not in panels:
            panels[pnl] = iface.new_panel(pnl, default_closed=(pnl != "Audio"))
        it = iface.new_socket(nm, in_out='INPUT', socket_type=st,
                              parent=panels[pnl])
        if desc:
            it.description = desc
        if dv is not None and hasattr(it, "default_value"):
            it.default_value = dv
        if lo is not None and hasattr(it, "min_value"):
            it.min_value = lo
        if hi is not None and hasattr(it, "max_value"):
            it.max_value = hi

    chladni = build_chladni_group()
    hfield = build_heightfield_group()
    sampler = build_sample_height_group()
    step = build_grain_step_group(chladni, sampler)
    audio = build_audio_group()
    table = build_mode_table_group()

    b = NB(tree)
    gi = b.node("NodeGroupInput")
    I = gi.outputs
    go = b.node("NodeGroupOutput")

    Lh = b.mul(I["Plate Size"], 0.5)
    # cells must be at least ~2.5 grain radii wide or one grain makes a spike
    G = b.mx(b.mn(I["Pile Resolution"],
                  b.m('FLOOR', b.add(b.div(I["Plate Size"], b.mul(I["Grain Radius"], 2.5)), 1.0))), 16.0)
    table_z = b.mul(I["Table Height"], -1.0)
    repose_tan = b.m('TANGENT', b.m('RADIANS', I["Angle of Repose"]))

    # ---------------- initial sprinkle --------------------------------
    pts = b.node("GeometryNodePoints")
    b.feed(pts.inputs["Count"], I["Grain Count"])
    idx = b.index()
    seed = I["Seed"]
    half_in = b.mul(Lh, b.sub(1.0, I["Sprinkle Margin"]))
    neg = b.mul(half_in, -1.0)
    rp = b.rand('FLOAT_VECTOR', b.xyz(neg, neg, 0.0), b.xyz(half_in, half_in, 0.0),
                idx, seed)
    b.feed(pts.inputs["Position"], rp)
    grains = pts.outputs["Points"]
    var = I["Size Variation"]
    # log-normal-ish radius distribution
    rr = b.rand('FLOAT', -1.0, 1.0, idx, b.imath('ADD', seed, 1))
    rr2 = b.rand('FLOAT', -1.0, 1.0, idx, b.imath('ADD', seed, 2))
    rmix = b.mul(b.add(rr, rr2), 0.5)
    rad = b.mul(I["Grain Radius"], b.m('EXPONENT', b.mul(rmix, b.mul(var, 1.4))))
    grains = b.store(grains, "cf_rad", rad)
    grains = b.store(grains, "cf_nest", b.rand('FLOAT', 0.0, 1.0, idx,
                                               b.imath('ADD', seed, 3)))
    grains = b.store(grains, "cf_tint", b.rand('FLOAT', 0.0, 1.0, idx,
                                               b.imath('ADD', seed, 4)))
    grains = b.store(grains, "cf_grip", b.m('POWER', b.rand(
        'FLOAT', 0.0, 1.0, idx, b.imath('ADD', seed, 8)), 30.0))
    grains = b.store(grains, "cf_rand", b.rand('FLOAT', 0.0, 1.0, idx,
                                               b.imath('ADD', seed, 5)))
    grains = b.store(grains, "cf_rot", b.rand('FLOAT_VECTOR', (0.0, 0.0, 0.0),
                                              (6.2832, 6.2832, 6.2832), idx,
                                              b.imath('ADD', seed, 6)),
                     'FLOAT_VECTOR')
    grains = b.store(grains, "cf_spin", b.rand('FLOAT_VECTOR', (-40.0, -40.0, -40.0),
                                               (40.0, 40.0, 40.0), idx,
                                               b.imath('ADD', seed, 7)),
                     'FLOAT_VECTOR')
    grains = b.store(grains, "cf_vel", (0.0, 0.0, 0.0), 'FLOAT_VECTOR')
    grains = b.store(grains, "cf_off", False, 'BOOLEAN')
    rz = b.mul(b.named("cf_rad"), 0.85)
    px0, py0, _ = b.sep(b.pos())
    grains = b.set_pos(grains, position=b.xyz(px0, py0, rz))

    def pile_of(geo):
        """Height field of the resting, on-plate grains of geo."""
        sel = b.AND(b.NOT(b.named("cf_off", 'BOOLEAN')),
                    b.lt(b.v('LENGTH', b.named("cf_vel", 'FLOAT_VECTOR')), 1e-5))
        sp = b.node("GeometryNodeSeparateGeometry", domain='POINT')
        b.feed(sp.inputs["Geometry"], geo)
        b.feed(sp.inputs["Selection"], sel)
        hf = b.group(hfield, **{"Grains": sp.outputs["Selection"],
                                "Half Size": Lh, "Resolution": G})
        return hf.outputs["Height Field"]

    # ---------------- simulation zone ---------------------------------
    sim_in = b.node("GeometryNodeSimulationInput")
    sim_out = b.node("GeometryNodeSimulationOutput")
    sim_in.pair_with_output(sim_out)
    for nm in ("Env", "Peak", "BassSlow", "BassPeak", "Mode", "Drive"):
        sim_out.state_items.new('FLOAT', nm)
    sim_out.state_items.new('GEOMETRY', "Pile")
    b.L.new(grains, sim_in.inputs["Geometry"])
    sim_in.inputs["Env"].default_value = 0.0
    sim_in.inputs["Peak"].default_value = 1e-3
    sim_in.inputs["BassPeak"].default_value = 1e-3
    b.L.new(b.add(I["Mode Min"], 0.0), sim_in.inputs["Mode"])

    b.L.new(pile_of(grains), sim_in.inputs["Pile"])

    S = sim_in.outputs
    dtf = S["Delta Time"]
    # audio time: playhead seconds aligned so frame 1 == t 0
    st = b.node("GeometryNodeInputSceneTime")
    secs = st.outputs["Seconds"]
    frame = st.outputs["Frame"]
    sim_out.state_items.new('FLOAT', "T0")
    b.L.new(secs, sim_in.inputs["T0"])
    b.L.new(S["T0"], sim_out.inputs["T0"])
    t_audio = b.add(b.sub(secs, S["T0"]), I["Audio Offset"])
    au = b.group(audio, **{"Sound": I["Sound"], "Time": t_audio,
                           "Low Hz": I["Pitch Low Hz"], "High Hz": I["Pitch High Hz"],
                           "Bass Low Hz": I["Bass Low Hz"],
                           "Bass High Hz": I["Bass High Hz"]})
    loud = au.outputs["Loudness"]
    bass = au.outputs["Bass"]
    pitch = au.outputs["Pitch"]

    def decay(tau):
        return b.m('EXPONENT', b.div(b.mul(dtf, -1.0), tau))

    # auto gain: slowly decaying peak follower
    peak = b.mx(b.mx(loud, b.mul(S["Peak"], decay(10.0))), 1e-4)
    level = b.clamp01(b.div(loud, peak))
    # envelope: fast attack, plate ring-down release
    env_prev = S["Env"]
    rising = b.gt(level, env_prev)
    tau = b.switch('FLOAT', rising, I["Ring Down"], 0.02)
    env = b.madd(b.sub(level, env_prev), b.sub(1.0, decay(tau)), env_prev)
    # bass transient
    bpeak = b.mx(b.mx(bass, b.mul(S["BassPeak"], decay(10.0))), 1e-4)
    # first frame: start the slow bass follower at the current bass level
    bs_prev = b.switch('FLOAT', b.gt(dtf, 0.0), bass, S["BassSlow"])
    bslow = b.madd(b.sub(bass, bs_prev), b.sub(1.0, decay(0.35)), bs_prev)
    punch = b.clamp01(b.div(b.mx(b.sub(bass, bslow), 0.0), bpeak))
    drive = b.madd(punch, I["Bass Punch"],
                   b.mul(b.m('POWER', env, 1.3), I["Drive"]))

    # resonance selection with hysteresis (holds through quiet passages)
    mmin = b.add(I["Mode Min"], 0.0)
    mmax = b.mx(b.add(I["Mode Max"], 0.0), mmin)
    target = b.madd(b.clamp01(pitch), b.sub(mmax, mmin), mmin)
    mode_prev = S["Mode"]
    far = b.gt(b.m('ABSOLUTE', b.sub(target, mode_prev)), I["Mode Hysteresis"])
    audible = b.gt(env, 0.15)
    mode = b.switch('FLOAT', b.AND(far, audible), mode_prev, b.m('ROUND', target))
    mode = b.mx(b.mn(mode, mmax), mmin)
    tbl = b.group(table, Mode=b.m('ROUND', mode))
    M, Nn, Sg = tbl.outputs["M"], tbl.outputs["N"], tbl.outputs["Sign"]

    # pile height field of the resting sand (carried in the sim state)
    geo_prev = S["Geometry"]
    hmesh = S["Pile"]

    # sub-stepping
    rin = b.node("GeometryNodeRepeatInput")
    rout = b.node("GeometryNodeRepeatOutput")
    rin.pair_with_output(rout)
    # Substeps are defined per 1/24 s so the time resolution (and cost per
    # second) is the same at any frame rate.
    b.feed(rin.inputs["Iterations"],
           b.mx(b.m('CEIL', b.mul(b.mul(I["Substeps"], dtf), 24.0)), 1.0))
    b.L.new(geo_prev, rin.inputs["Geometry"])
    sub_dt = b.div(dtf, b.mx(b.add(I["Substeps"], 0.0), 1.0))
    seed_i = b.imath('ADD', b.imath('MULTIPLY', b.m('ROUND', frame), 131),
                     rin.outputs["Iteration"])
    stp = b.group(step, **{
        "Grains": rin.outputs["Geometry"], "Height Field": hmesh,
        "dt": sub_dt, "Seed": seed_i, "Drive": drive, "M": M, "N": Nn,
        "Sign": Sg, "Half Size": Lh, "Resolution": G, "Table Z": table_z,
        "Gravity": I["Gravity"], "Hop Height": I["Hop Height"],
        "Launch Rate": I["Launch Rate"], "Scatter": I["Scatter"],
        "Node Attraction": I["Node Attraction"],
        "Restitution": I["Restitution"], "Friction": I["Friction"],
        "Repose Tan": repose_tan, "Creep": I["Creep"],
        "Avalanche Speed": b.mul(I["Grain Radius"], 150.0)})
    b.L.new(stp.outputs[0], rout.inputs["Geometry"])

    b.L.new(rout.outputs["Geometry"], sim_out.inputs["Geometry"])
    b.L.new(pile_of(rout.outputs["Geometry"]), sim_out.inputs["Pile"])
    b.L.new(env, sim_out.inputs["Env"])
    b.L.new(peak, sim_out.inputs["Peak"])
    b.L.new(bslow, sim_out.inputs["BassSlow"])
    b.L.new(bpeak, sim_out.inputs["BassPeak"])
    b.L.new(mode, sim_out.inputs["Mode"])
    b.L.new(drive, sim_out.inputs["Drive"])

    # ---------------- output: pile body + grains ----------------------
    grains_out = sim_out.outputs["Geometry"]
    O = sim_out.outputs
    # expose the drive for the shader / debugging
    grains_out = b.store(grains_out, "cf_drive", O["Drive"])
    grains_out = b.store(grains_out, "cf_mode", b.m('ROUND', O["Mode"]))

    # pile body sits under the grain skin; areas without sand sink into
    # the plate (hidden).
    pile = O["Pile"]
    hp = b.named("cf_h")
    ppx, ppy, _ = b.sep(b.pos())
    # A continuous surface (no trimmed cells -> no saw-tooth edges): where
    # there is little sand it dips smoothly below the plate top and is hidden
    # inside the 1.5 mm plate.
    body_z = b.mx(b.sub(hp, b.mul(I["Grain Radius"], 2.4)), -0.0009)
    pile = b.set_pos(pile, position=b.xyz(ppx, ppy, body_z))
    smooth = b.node("GeometryNodeSetShadeSmooth")
    b.feed(smooth.inputs["Geometry"], pile)
    pile = smooth.outputs[0]
    pm = b.node("GeometryNodeSetMaterial")
    b.feed(pm.inputs["Geometry"], pile)
    b.feed(pm.inputs["Material"], I["Pile Material"])
    pile = pm.outputs[0]

    # grains: instances of real grain shapes (render) or points (viewport)
    shapes = b.node("GeometryNodeGroup")
    shapes.node_tree = grain_shapes_tree
    shp = shapes.outputs[0]
    shp_mat = b.node("GeometryNodeSetMaterial")
    b.feed(shp_mat.inputs["Geometry"], shp)
    b.feed(shp_mat.inputs["Material"], I["Grain Material"])
    shp = shp_mat.outputs[0]

    count_inst = b.node("GeometryNodeAttributeDomainSize", component='INSTANCES')
    b.feed(count_inst.inputs["Geometry"], shp)
    n_inst = count_inst.outputs["Instance Count"]
    iop = b.node("GeometryNodeInstanceOnPoints")
    b.feed(iop.inputs["Points"], grains_out)
    b.feed(iop.inputs["Instance"], shp)
    iop.inputs["Pick Instance"].default_value = True
    pick = b.rand('INT', 0, b.imath('SUBTRACT', n_inst, 1),
                  b.node("GeometryNodeInputID").outputs[0], 17)
    b.feed(iop.inputs["Instance Index"], pick)
    b.feed(iop.inputs["Rotation"], b.named("cf_rot", 'FLOAT_VECTOR'))
    rs = b.named("cf_rad")
    # slightly flattened, elongated grains
    sx = b.rand('FLOAT', 0.85, 1.25, b.node("GeometryNodeInputID").outputs[0], 23)
    b.feed(iop.inputs["Scale"], b.v('SCALE', b.xyz(sx, 1.0, 0.8), scale=rs))
    inst = iop.outputs["Instances"]

    # viewport proxy: points with radius
    pr = b.node("GeometryNodeSetPointRadius")
    b.feed(pr.inputs["Points"], grains_out)
    b.feed(pr.inputs["Radius"], b.named("cf_rad"))
    pmat = b.node("GeometryNodeSetMaterial")
    b.feed(pmat.inputs["Geometry"], pr.outputs[0])
    b.feed(pmat.inputs["Material"], I["Grain Material"])
    proxy = pmat.outputs[0]

    isvp = b.node("GeometryNodeIsViewport").outputs[0]
    use_proxy = b.AND(isvp, b.NOT(I["Viewport Full Grains"]))
    grains_vis = b.switch('GEOMETRY', use_proxy, inst, proxy)

    join = b.node("GeometryNodeJoinGeometry")
    b.L.new(grains_vis, join.inputs[0])
    b.L.new(pile, join.inputs[0])
    b.L.new(join.outputs[0], go.inputs[0])
    _layout(tree, dx=260, dy=190)
    return tree


# ======================================================================
#  LOOK: grain shapes, materials, studio apparatus, lights, cameras
#  (fully procedural - no image textures or HDRIs)
# ======================================================================

BUMPD = 0.04  # Bump distance acts in instance object space (grain radius == 1)


# ---------------------------------------------------------------------------
# small node helpers
# ---------------------------------------------------------------------------

def _clear(nt):
    for n in list(nt.nodes):
        nt.nodes.remove(n)


def _n(nt, typ, x=0, y=0, label=None, **props):
    n = nt.nodes.new(typ)
    n.location = (x, y)
    if label:
        n.label = label
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _set(sock, val):
    sock.default_value = val


def _feed(nt, sock, val):
    """Connect a socket or set a constant."""
    if isinstance(val, bpy.types.NodeSocket):
        nt.links.new(val, sock)
    elif val is not None:
        sock.default_value = val


def _math(nt, op, a=None, b=None, c=None, x=0, y=0, clamp=False):
    n = _n(nt, "ShaderNodeMath", x, y, operation=op, use_clamp=clamp)
    _feed(nt, n.inputs[0], a)
    _feed(nt, n.inputs[1], b)
    if c is not None:
        _feed(nt, n.inputs[2], c)
    return n.outputs[0]


def _vmath(nt, op, a=None, b=None, x=0, y=0, scale=None):
    n = _n(nt, "ShaderNodeVectorMath", x, y, operation=op)
    _feed(nt, n.inputs[0], a)
    _feed(nt, n.inputs[1], b)
    if scale is not None:
        _feed(nt, n.inputs["Scale"], scale)
    out = n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]
    return out


def _mix(nt, fac, a, b, data_type="RGBA", blend="MIX", x=0, y=0, clamp=True):
    n = _n(nt, "ShaderNodeMix", x, y, data_type=data_type)
    if data_type == "RGBA":
        n.blend_type = blend
        n.clamp_result = clamp
        ia, ib, o = n.inputs[6], n.inputs[7], n.outputs[2]
    elif data_type == "FLOAT":
        ia, ib, o = n.inputs[2], n.inputs[3], n.outputs[0]
    else:  # VECTOR
        ia, ib, o = n.inputs[4], n.inputs[5], n.outputs[1]
    _feed(nt, n.inputs[0], fac)
    _feed(nt, ia, a)
    _feed(nt, ib, b)
    return o


def _maprange(nt, v, a, b, c, d, x=0, y=0, interp="LINEAR", clamp=True):
    n = _n(nt, "ShaderNodeMapRange", x, y, interpolation_type=interp, clamp=clamp)
    _feed(nt, n.inputs[0], v)
    n.inputs[1].default_value = a
    n.inputs[2].default_value = b
    n.inputs[3].default_value = c
    n.inputs[4].default_value = d
    return n.outputs[0]


def _noise(nt, vec, scale, detail=2.0, rough=0.5, dist=0.0, x=0, y=0, dims="3D", w=None, ntype=None):
    n = _n(nt, "ShaderNodeTexNoise", x, y, noise_dimensions=dims)
    if ntype:
        n.noise_type = ntype
    _feed(nt, n.inputs["Vector"], vec)
    _feed(nt, n.inputs["Scale"], scale)
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = dist
    if w is not None:
        _feed(nt, n.inputs["W"], w)
    return n


def _ramp(nt, fac, stops, interp="LINEAR", x=0, y=0):
    """stops: list of (pos, (r,g,b,a))"""
    n = _n(nt, "ShaderNodeValToRGB", x, y)
    cr = n.color_ramp
    cr.interpolation = interp
    while len(cr.elements) > 1:
        cr.elements.remove(cr.elements[-1])
    cr.elements[0].position = stops[0][0]
    cr.elements[0].color = stops[0][1]
    for p, c in stops[1:]:
        e = cr.elements.new(p)
        e.color = c
    _feed(nt, n.inputs[0], fac)
    return n.outputs[0]


def _new_material(name):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True  # deprecated in 5.x but harmless
    except Exception:
        pass
    _clear(m.node_tree)
    return m


def _gray(v):
    return (v, v, v, 1.0)


# ---------------------------------------------------------------------------
# 1. grain shapes
# ---------------------------------------------------------------------------

def _gn_new(name):
    t = bpy.data.node_groups.get(name)
    if t is None:
        t = bpy.data.node_groups.new(name, "GeometryNodeTree")
    else:  # rebuild in place so existing users (modifiers / group nodes) stay linked
        _clear(t)
        t.interface.clear()
    t.interface.new_socket("Instances", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return t


def _gn_noise_disp(nt, geo, x, y, offset, scale, amp, detail=2.0, rough=0.5):
    """displace along the radial direction by (noise-0.5)*amp."""
    pos = _n(nt, "GeometryNodeInputPosition", x, y - 200).outputs[0]
    p2 = _vmath(nt, "ADD", pos, offset, x + 150, y - 200)
    nz = _noise(nt, p2, scale, detail, rough, x=x + 300, y=y - 200)
    d = _math(nt, "SUBTRACT", nz.outputs["Factor"], 0.5, x=x + 450, y=y - 200)
    d = _math(nt, "MULTIPLY", d, amp, x=x + 600, y=y - 200)
    nrm = _vmath(nt, "NORMALIZE", pos, None, x + 450, y - 350)
    off = _vmath(nt, "SCALE", nrm, None, x + 750, y - 300, scale=d)
    sp = _n(nt, "GeometryNodeSetPosition", x + 900, y)
    nt.links.new(geo, sp.inputs["Geometry"])
    nt.links.new(off, sp.inputs["Offset"])
    return sp.outputs[0]


def _gn_clip_plane(nt, geo, normal, d, x, y):
    """p -= n * max(dot(p,n)-d, 0)   -> flat facet."""
    pos = _n(nt, "GeometryNodeInputPosition", x, y - 200).outputs[0]
    dp = _vmath(nt, "DOT_PRODUCT", pos, normal, x + 150, y - 200)
    e = _math(nt, "SUBTRACT", dp, d, x=x + 300, y=y - 200)
    e = _math(nt, "MAXIMUM", e, 0.0, x=x + 450, y=y - 200)
    e = _math(nt, "MULTIPLY", e, -1.0, x=x + 600, y=y - 200)
    off = _vmath(nt, "SCALE", normal, None, x + 750, y - 200, scale=e)
    sp = _n(nt, "GeometryNodeSetPosition", x + 900, y)
    nt.links.new(geo, sp.inputs["Geometry"])
    nt.links.new(off, sp.inputs["Offset"])
    return sp.outputs[0]


def _rand_unit(rng):
    while True:
        v = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
        l = math.sqrt(sum(c * c for c in v))
        if 0.2 < l <= 1.0:
            return tuple(c / l for c in v)


def build_grain_shapes(kind="SAND"):
    """GN group with no inputs, output 'Instances': 8 grain shapes, mean radius ~1."""
    kind = (kind or "SAND").upper()
    nt = _gn_new(PREFIX + "GrainShapes_" + kind)
    rng = random.Random(1234 if kind == "SAND" else 777)
    # grains carry their material so instancing needs no extra Set Material
    mat = bpy.data.materials.get(PREFIX + "Grain_" + kind) or make_grain_material(kind)
    count = 8
    g2i = _n(nt, "GeometryNodeGeometryToInstance", 4200, 0)
    out = _n(nt, "NodeGroupOutput", 4450, 0)
    nt.links.new(g2i.outputs[0], out.inputs[0])

    for i in range(count):
        y = -i * 900
        if kind == "SALT":
            cube = _n(nt, "GeometryNodeMeshCube", 0, y)
            cube.inputs["Size"].default_value = (1.62, 1.62, 1.62)
            sub = _n(nt, "GeometryNodeSubdivisionSurface", 200, y)
            sub.inputs["Level"].default_value = 2
            sub.inputs["Edge Crease"].default_value = rng.uniform(0.72, 0.86)
            nt.links.new(cube.outputs[0], sub.inputs[0])
            geo = sub.outputs[0]
            # slight warping / uneven faces
            geo = _gn_noise_disp(nt, geo, 400, y, (rng.uniform(-50, 50),) * 3, 1.4, 0.04, 1.0)
            # broken corners / chipped edges
            for k in range(rng.randint(1, 3)):
                c = [rng.choice((-1, 1)) for _ in range(3)]
                if rng.random() < 0.5:
                    c[rng.randrange(3)] = 0.0
                l = math.sqrt(sum(v * v for v in c))
                nrm = tuple(v / l + rng.uniform(-0.15, 0.15) for v in c)
                l = math.sqrt(sum(v * v for v in nrm))
                nrm = tuple(v / l for v in nrm)
                geo = _gn_clip_plane(nt, geo, nrm, rng.uniform(0.62, 0.85), 1400 + k * 1100, y)
            sx = (1.0, rng.uniform(0.82, 1.0), rng.uniform(0.62, 0.95))
            norm = (sx[0] * sx[1] * sx[2]) ** (-1.0 / 3.0) * 0.98
        else:
            ico = _n(nt, "GeometryNodeMeshIcoSphere", 0, y)
            ico.inputs["Radius"].default_value = 1.0
            ico.inputs["Subdivisions"].default_value = 3  # 320 tris
            geo = ico.outputs["Mesh"]
            o = (rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100))
            # big lumpy form + finer pitting
            geo = _gn_noise_disp(nt, geo, 200, y, o, rng.uniform(0.7, 1.1), rng.uniform(0.25, 0.42), 1.0, 0.4)
            geo = _gn_noise_disp(nt, geo, 1200, y, (o[1], o[2], o[0]), 3.2, 0.09, 2.0, 0.55)
            # sub-angular: 2-4 broken flat facets, rounded by smooth shading
            nf = rng.randint(3, 6)
            dmin = 1.0
            for k in range(nf):
                d = rng.uniform(0.58, 0.84)
                dmin = min(dmin, d)
                geo = _gn_clip_plane(nt, geo, _rand_unit(rng), d, 2200 + k * 1000, y)
            # conchoidal, not saw-cut: break the planarity of the facets a little
            geo = _gn_noise_disp(nt, geo, 2200 + nf * 1000, y, (o[2], o[0], o[1]), 2.2, 0.06, 1.0, 0.5)
            a = rng.uniform(1.0, 1.35)
            b = rng.uniform(0.85, 1.1)
            c = rng.uniform(0.6, 0.85)
            sx = (a, b, c)
            norm = (a * b * c) ** (-1.0 / 3.0) * 1.16  # compensate volume lost to facets
        xf = _n(nt, "GeometryNodeTransform", 3700, y)
        xf.inputs["Scale"].default_value = tuple(s * norm for s in sx)
        nt.links.new(geo, xf.inputs[0])
        ss = _n(nt, "GeometryNodeSetShadeSmooth", 3900, y)
        nt.links.new(xf.outputs[0], ss.inputs[0])
        sm = _n(nt, "GeometryNodeSetMaterial", 4050, y)
        sm.inputs["Material"].default_value = mat
        nt.links.new(ss.outputs[0], sm.inputs[0])
        nt.links.new(sm.outputs[0], g2i.inputs[0])
    return nt


# ---------------------------------------------------------------------------
# 2/3. grain + fill materials (shared palette group)
# ---------------------------------------------------------------------------

def _palette_group(kind):
    """ShaderNodeTree group: in Tint, Rand  ->  Color, Roughness, SSS, Metallic, Glint."""
    name = PREFIX + "GrainPalette_" + kind
    g = bpy.data.node_groups.get(name)
    if g is not None and getattr(g, "get", None) and g.get("cf_built"):
        return g  # shared by grain + fill materials: never delete (would orphan users)
    if g is None:
        g = bpy.data.node_groups.new(name, "ShaderNodeTree")
    _clear(g)
    g.interface.clear()
    g["cf_built"] = 1
    it = g.interface
    it.new_socket("Tint", in_out="INPUT", socket_type="NodeSocketFloat")
    it.new_socket("Rand", in_out="INPUT", socket_type="NodeSocketFloat")
    for nm, st in (("Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat"),
                   ("SSS", "NodeSocketFloat"), ("Metallic", "NodeSocketFloat"),
                   ("Glint", "NodeSocketFloat"), ("Clear", "NodeSocketFloat")):
        it.new_socket(nm, in_out="OUTPUT", socket_type=st)
    gi = _n(g, "NodeGroupInput", -900, 0)
    go = _n(g, "NodeGroupOutput", 900, 0)
    tint, rnd = gi.outputs["Tint"], gi.outputs["Rand"]
    # decorrelated randoms
    r2 = _math(g, "FRACT", _math(g, "MULTIPLY", rnd, 7.317, x=-700, y=-300), x=-550, y=-300)
    r3 = _math(g, "FRACT", _math(g, "MULTIPLY", rnd, 13.71, x=-700, y=-450), x=-550, y=-450)

    if kind == "SALT":
        col = _ramp(g, tint, [
            (0.0, (0.80, 0.80, 0.78, 1)),
            (0.55, (0.86, 0.86, 0.85, 1)),
            (0.85, (0.78, 0.76, 0.72, 1)),
            (0.96, (0.62, 0.60, 0.57, 1)),
        ], "LINEAR", -300, 300)
        rough = _maprange(g, r2, 0, 1, 0.05, 0.25, -300, 0)
        sss = _maprange(g, r3, 0, 1, 0.75, 1.0, -300, -150)
        met = 0.0
        glint = _math(g, "GREATER_THAN", r2, 0.7, x=-300, y=-300)
        clear = 0.6
    else:
        # mineral classes (constant ramp on tint): fractions of a typical
        # beach / play sand: mostly quartz (clear, milky, tan, iron-stained),
        # feldspar (pink/cream), lithic grey and dark heavy minerals.
        col = _ramp(g, tint, [
            (0.00, (0.70, 0.62, 0.49, 1)),   # cream silica
            (0.30, (0.74, 0.69, 0.60, 1)),   # milky / frosted quartz
            (0.52, (0.62, 0.58, 0.51, 1)),   # clear quartz (darker: see-through)
            (0.64, (0.66, 0.56, 0.41, 1)),   # light tan
            (0.86, (0.72, 0.66, 0.56, 1)),   # off-white
            (0.94, (0.55, 0.43, 0.29, 1)),   # rare amber, iron-stained
            (0.978, (0.09, 0.085, 0.08, 1)),  # rare dark heavy mineral
        ], "CONSTANT", -300, 300)
        # per-grain value/saturation jitter (kept low: demo sand is uniform)
        hsv = _n(g, "ShaderNodeHueSaturation", 0, 300)
        _feed(g, hsv.inputs["Hue"], _maprange(g, r3, 0, 1, 0.49, 0.51, -150, 450))
        _feed(g, hsv.inputs["Saturation"], _maprange(g, r2, 0, 1, 0.85, 1.15, -150, 150))
        _feed(g, hsv.inputs["Value"], _maprange(g, rnd, 0, 1, 0.86, 1.14, -150, 0))
        g.links.new(col, hsv.inputs["Color"])
        col = hsv.outputs[0]
        dark = _math(g, "GREATER_THAN", tint, 0.978, x=-300, y=-600)
        clear = _math(g, "MULTIPLY",
                      _math(g, "GREATER_THAN", tint, 0.52, x=-500, y=-750),
                      _math(g, "LESS_THAN", tint, 0.64, x=-500, y=-850), x=-300, y=-750)
        rough = _maprange(g, r2, 0, 1, 0.32, 0.6, -300, 0)
        # clear quartz is glossier, occasional fresh-fracture grain sparkles
        glint = _math(g, "MAXIMUM", _math(g, "GREATER_THAN", r3, 0.86, x=-300, y=-900), clear, x=-150, y=-900)
        rough = _mix(g, glint, rough, _maprange(g, r2, 0, 1, 0.06, 0.16, -150, -1050), "FLOAT", x=0, y=0)
        sss = _mix(g, dark, _maprange(g, r3, 0, 1, 0.2, 0.45, -150, -150), 0.0, "FLOAT", x=0, y=-150)
        sss = _math(g, "MAXIMUM", sss, _math(g, "MULTIPLY", clear, 0.7, x=0, y=-300), x=150, y=-150)
        met = _math(g, "MULTIPLY", dark, 0.35, x=0, y=-600)
    g.links.new(col, go.inputs["Color"])
    _feed(g, go.inputs["Roughness"], rough)
    _feed(g, go.inputs["SSS"], sss)
    _feed(g, go.inputs["Metallic"], met)
    _feed(g, go.inputs["Glint"], glint)
    _feed(g, go.inputs["Clear"], clear)
    return g


def _grain_bsdf(nt, pal, x, y, ss_radius_m):
    b = _n(nt, "ShaderNodeBsdfPrincipled", x, y)
    b.subsurface_method = "RANDOM_WALK"
    nt.links.new(pal.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(pal.outputs["Roughness"], b.inputs["Roughness"])
    nt.links.new(pal.outputs["Metallic"], b.inputs["Metallic"])
    nt.links.new(pal.outputs["SSS"], b.inputs["Subsurface Weight"])
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.75, 0.5)
    b.inputs["Subsurface Scale"].default_value = ss_radius_m
    b.inputs["IOR"].default_value = 1.54
    return b


def make_grain_material(kind="SAND"):
    kind = (kind or "SAND").upper()
    m = _new_material(PREFIX + "Grain_" + kind)
    nt = m.node_tree
    a_t = _n(nt, "ShaderNodeAttribute", -1200, 200, attribute_name="cf_tint", attribute_type="INSTANCER")
    a_r = _n(nt, "ShaderNodeAttribute", -1200, 0, attribute_name="cf_rand", attribute_type="INSTANCER")
    pal = _n(nt, "ShaderNodeGroup", -900, 100)
    pal.node_tree = _palette_group(kind)
    nt.links.new(a_t.outputs["Factor"], pal.inputs["Tint"])
    nt.links.new(a_r.outputs["Factor"], pal.inputs["Rand"])
    b = _grain_bsdf(nt, pal, -300, 200, 0.00018 if kind == "SAND" else 0.0004)
    # glassy grains (clear quartz / rock salt) partly transmit
    _feed(nt, b.inputs["Transmission Weight"],
          _math(nt, "MULTIPLY", pal.outputs["Clear"], 1.0 if kind == "SALT" else 0.55, x=-500, y=500))
    if kind == "SALT":
        b.inputs["IOR"].default_value = 1.544
    # surface micro texture in grain-local space (grain radius == 1 in object space)
    tc = _n(nt, "ShaderNodeTexCoord", -1200, -300)
    off = _vmath(nt, "SCALE", _n(nt, "ShaderNodeCombineXYZ", -1050, -450).outputs[0], None, -900, -450,
                 scale=_math(nt, "MULTIPLY", a_r.outputs["Factor"], 100.0, x=-1050, y=-600))
    p = _vmath(nt, "ADD", tc.outputs["Object"], off, -750, -300)
    if kind == "SALT":
        # stepped / hopper-crystal terraces + frosting
        nz = _noise(nt, p, 9.0, 3.0, 0.6, x=-600, y=-300)
        bump_h = nz.outputs["Factor"]
        strength = 0.3
    else:
        nz = _noise(nt, p, 7.0, 4.0, 0.65, x=-600, y=-300)
        vor = _n(nt, "ShaderNodeTexVoronoi", -600, -550, feature="F1")
        nt.links.new(p, vor.inputs["Vector"])
        vor.inputs["Scale"].default_value = 5.0
        bump_h = _math(nt, "ADD", nz.outputs["Factor"],
                       _math(nt, "MULTIPLY", vor.outputs["Distance"], 0.35, x=-450, y=-550), x=-300, y=-400)
        strength = 1.0
    bump = _n(nt, "ShaderNodeBump", -150, -300)
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = BUMPD
    nt.links.new(bump_h, bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    # frosted/pitted grains: modulate roughness a little by the micro texture
    r = _math(nt, "ADD", pal.outputs["Roughness"],
              _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", nz.outputs["Factor"], 0.5, x=-450, y=-800), 0.25,
                    x=-300, y=-800), x=-150, y=-700, clamp=True)
    nt.links.new(r, b.inputs["Roughness"])
    o = _n(nt, "ShaderNodeOutputMaterial", 200, 200)
    nt.links.new(b.outputs[0], o.inputs["Surface"])
    return m


def make_fill_material(kind="SAND", grain_radius=None):
    """Heightfield 'body' under the grains: granular at ~0.7 mm, same palette."""
    kind = (kind or "SAND").upper()
    m = _new_material(PREFIX + "Fill_" + kind)
    nt = m.node_tree
    tc = _n(nt, "ShaderNodeTexCoord", -1400, 0)
    pos = tc.outputs["Object"]
    cell = 0.00072 if kind == "SAND" else 0.0010
    if grain_radius:
        cell = grain_radius * (2.0 if kind == "SAND" else 2.8)
    vor = _n(nt, "ShaderNodeTexVoronoi", -1150, 0, feature="F1", voronoi_dimensions="3D")
    nt.links.new(pos, vor.inputs["Vector"])
    vor.inputs["Scale"].default_value = 1.0 / cell
    vor.inputs["Randomness"].default_value = 1.0
    sep = _n(nt, "ShaderNodeSeparateColor", -950, 0)
    nt.links.new(vor.outputs["Color"], sep.inputs[0])
    pal = _n(nt, "ShaderNodeGroup", -750, 0)
    pal.node_tree = _palette_group(kind)
    nt.links.new(sep.outputs[0], pal.inputs["Tint"])
    nt.links.new(sep.outputs[1], pal.inputs["Rand"])
    # pile thickness -> occlusion (gaps between surface grains look into shadowed depth)
    ah = _n(nt, "ShaderNodeAttribute", -1150, -400, attribute_name="cf_h", attribute_type="GEOMETRY")
    occ = _maprange(nt, ah.outputs["Factor"], 0.0, 0.0018, 0.28, 0.6, -950, -400, interp="SMOOTHSTEP")
    # darken toward cell borders (the crevices between grains)
    crev = _maprange(nt, vor.outputs["Distance"], 0.1, 0.5, 1.0, 0.22, -950, -250, interp="SMOOTHSTEP")
    shade = _math(nt, "MULTIPLY", occ, crev, x=-750, y=-300)
    col = _mix(nt, 1.0, pal.outputs["Color"], shade, "RGBA", "MULTIPLY", -450, 100)
    b = _n(nt, "ShaderNodeBsdfPrincipled", -150, 100)
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.6
    _feed(nt, b.inputs["Subsurface Weight"], _math(nt, "MULTIPLY", pal.outputs["SSS"], 0.4, x=-450, y=-100))
    b.inputs["Subsurface Scale"].default_value = 0.0003
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.75, 0.5)
    # domed grains bump
    dome = _math(nt, "POWER", _math(nt, "SUBTRACT", 1.0, vor.outputs["Distance"], x=-950, y=-600, clamp=True),
                 0.5, x=-750, y=-600)
    bump = _n(nt, "ShaderNodeBump", -450, -500)
    bump.inputs["Strength"].default_value = 0.8
    bump.inputs["Distance"].default_value = cell * 0.3
    nt.links.new(dome, bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    o = _n(nt, "ShaderNodeOutputMaterial", 150, 100)
    nt.links.new(b.outputs[0], o.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# 4. plate material
# ---------------------------------------------------------------------------

def _scratch_layer(nt, pos, scale, density, seed, x, y, angle_deg=0.0):
    """Fine straight micro-scratches: very stretched noise, thresholded."""
    a = math.radians(angle_deg)
    mp = _n(nt, "ShaderNodeMapping", x, y)
    mp.inputs["Rotation"].default_value = (0, 0, a)
    mp.inputs["Location"].default_value = (seed, seed * 0.37, seed * 0.11)
    mp.inputs["Scale"].default_value = (scale * 0.006, scale, scale)
    nt.links.new(pos, mp.inputs["Vector"])
    nz = _noise(nt, mp.outputs[0], 1.0, 1.0, 0.5, x=x + 200, y=y)
    # thin lines where noise crosses 0.5
    d = _math(nt, "ABSOLUTE", _math(nt, "SUBTRACT", nz.outputs["Factor"], 0.5, x=x + 350, y=y), x=x + 500, y=y)
    line = _maprange(nt, d, 0.0, 0.006, 1.0, 0.0, x + 650, y)
    # sparse: mask with low-freq noise
    msk = _noise(nt, pos, 18.0, 1.0, 0.5, x=x + 350, y=y - 250, w=None)
    mk = _maprange(nt, msk.outputs["Factor"], 1.0 - density, 1.0 - density + 0.06, 0.0, 1.0, x + 500, y - 250)
    return _math(nt, "MULTIPLY", line, mk, x=x + 800, y=y)


def make_plate_material(finish="BLACK", plate_size=0.30):
    finish = (finish or "BLACK").upper()
    m = _new_material(PREFIX + "Plate_" + finish)
    nt = m.node_tree
    tc = _n(nt, "ShaderNodeTexCoord", -2400, 0)
    pos = tc.outputs["Object"]  # metres, plate centred at origin

    # ---- imperfection masks -------------------------------------------------
    s1 = _scratch_layer(nt, pos, 900.0, 0.35, 3.1, -2100, 800, 12.0)
    s2 = _scratch_layer(nt, pos, 1300.0, 0.30, 7.7, -2100, 400, -37.0)
    s3 = _scratch_layer(nt, pos, 700.0, 0.25, 1.9, -2100, 0, 71.0)
    scratches = _math(nt, "MAXIMUM", _math(nt, "MAXIMUM", s1, s2, x=-1150, y=600), s3, x=-1000, y=500)

    # fingerprints: a few elliptical patches of concentric distorted ridges
    fp_total = None
    k_s = plate_size / 0.30
    prints = [(-0.105 * k_s, -0.095 * k_s, 0.0, 0.009), (0.112 * k_s, 0.118 * k_s, 1.2, 0.010),
              (0.118 * k_s, -0.085 * k_s, 2.4, 0.008), (-0.06 * k_s, 0.125 * k_s, 0.6, 0.0085)]
    for k, (cx, cy, rot, rad) in enumerate(prints):
        yy = -400 - k * 300
        mp = _n(nt, "ShaderNodeMapping", -2100, yy)
        mp.inputs["Location"].default_value = (cx, cy, 0)
        mp.inputs["Rotation"].default_value = (0, 0, rot)
        mp.inputs["Scale"].default_value = (1.0, 1.35, 1.0)
        mp.vector_type = "TEXTURE"
        nt.links.new(pos, mp.inputs["Vector"])
        r = _vmath(nt, "LENGTH", _vmath(nt, "MULTIPLY", mp.outputs[0], (1, 1, 0), -1900, yy), None, -1750, yy)
        blob = _maprange(nt, r, rad * 0.55, rad, 1.0, 0.0, -1600, yy, interp="SMOOTHSTEP")
        wv = _n(nt, "ShaderNodeTexWave", -1600, yy - 150, wave_type="RINGS", rings_direction="SPHERICAL")
        nt.links.new(mp.outputs[0], wv.inputs["Vector"])
        wv.inputs["Scale"].default_value = 700.0  # rings period 2pi/20/scale ~0.45 mm
        wv.inputs["Distortion"].default_value = 2.5
        wv.inputs["Detail"].default_value = 1.5
        wv.inputs["Detail Scale"].default_value = 1.2
        rid = _maprange(nt, wv.outputs["Factor"], 0.35, 0.75, 0.0, 1.0, -1450, yy - 150)
        fp = _math(nt, "MULTIPLY", blob, _math(nt, "ADD", rid, 0.35, x=-1300, y=yy - 150), x=-1150, y=yy)
        fp_total = fp if fp_total is None else _math(nt, "MAXIMUM", fp_total, fp, x=-1000, y=yy)
    # broad smudges / wipe marks
    smg = _noise(nt, pos, 9.0, 3.0, 0.6, dist=0.4, x=-1600, y=-1700)
    smudge = _maprange(nt, smg.outputs["Factor"], 0.58, 0.75, 0.0, 1.0, -1450, -1700)
    smear = _math(nt, "MAXIMUM", _math(nt, "MULTIPLY", fp_total, 0.8, x=-900, y=-900),
                  _math(nt, "MULTIPLY", smudge, 0.25, x=-1300, y=-1700), x=-800, y=-1000)

    # dust: sparse tiny specks (voronoi cells, tiny threshold) + fibres
    vd = _n(nt, "ShaderNodeTexVoronoi", -1600, -2000, feature="F1", voronoi_dimensions="3D")
    nt.links.new(pos, vd.inputs["Vector"])
    vd.inputs["Scale"].default_value = 260.0
    sep = _n(nt, "ShaderNodeSeparateColor", -1450, -2150)
    nt.links.new(vd.outputs["Color"], sep.inputs[0])
    speck_r = _math(nt, "MULTIPLY", _math(nt, "POWER", sep.outputs[0], 3.0, x=-1300, y=-2150), 0.1, x=-1150, y=-2150)
    dust = _math(nt, "LESS_THAN", vd.outputs["Distance"], speck_r, x=-1000, y=-2000)
    dust = _math(nt, "MULTIPLY", dust, _math(nt, "GREATER_THAN", sep.outputs[1], 0.8, x=-1150, y=-1900),
                 x=-950, y=-1950)
    dmask = _noise(nt, pos, 6.0, 2.0, 0.5, x=-1300, y=-2350)
    dust = _math(nt, "MULTIPLY", dust,
                 _maprange(nt, dmask.outputs["Factor"], 0.4, 0.6, 0.25, 1.0, -1150, -2350), x=-850, y=-2050)

    # edge wear: distance to plate rim (square, 0.15 half-size) + chamfer normals
    ab = _vmath(nt, "ABSOLUTE", pos, None, -2100, -2600)
    sxyz = _n(nt, "ShaderNodeSeparateXYZ", -1950, -2600)
    nt.links.new(ab, sxyz.inputs[0])
    edge_d = _math(nt, "SUBTRACT", plate_size / 2, _math(nt, "MAXIMUM", sxyz.outputs[0], sxyz.outputs[1], x=-1800, y=-2600),
                   x=-1650, y=-2600)
    geo = _n(nt, "ShaderNodeNewGeometry", -2100, -2850)
    sepn = _n(nt, "ShaderNodeSeparateXYZ", -1950, -2850)
    nt.links.new(geo.outputs["Normal"], sepn.inputs[0])
    side = _maprange(nt, _math(nt, "ABSOLUTE", sepn.outputs[2], x=-1800, y=-2850), 0.95, 0.75, 0.0, 1.0, -1650, -2850)
    ewn = _noise(nt, pos, 900.0, 3.0, 0.7, x=-1650, y=-3050)
    rim = _maprange(nt, edge_d, 0.0, 0.0012, 1.0, 0.0, -1500, -2600)
    edgewear = _math(nt, "MAXIMUM", rim, side, x=-1350, y=-2700)
    edgewear = _math(nt, "MULTIPLY", edgewear,
                     _maprange(nt, ewn.outputs["Factor"], 0.42, 0.6, 0.0, 1.0, -1500, -3050), x=-1200, y=-2800)
    # large-scale roughness variation (handling, uneven anodize)
    big = _noise(nt, pos, 4.0, 3.0, 0.55, x=-1600, y=1200)
    bigv = _math(nt, "SUBTRACT", big.outputs["Factor"], 0.5, x=-1450, y=1200)

    b = _n(nt, "ShaderNodeBsdfPrincipled", 200, 200)
    if finish == "ALUMINUM":
        base = (0.80, 0.81, 0.82, 1)
        r0, r_scr, r_smear = 0.22, 0.12, 0.06
        b.inputs["Metallic"].default_value = 1.0
        # brushed along X: anisotropy + streaky roughness
        mp = _n(nt, "ShaderNodeMapping", -1600, 1500)
        mp.inputs["Scale"].default_value = (2.0, 2500.0, 40.0)
        nt.links.new(pos, mp.inputs["Vector"])
        br = _noise(nt, mp.outputs[0], 1.0, 4.0, 0.6, x=-1450, y=1500)
        brush = br.outputs["Factor"]
        b.inputs["Anisotropic"].default_value = 0.75
        # brushing runs along object X; plate is never rotated so a constant world tangent works
        tg = _n(nt, "ShaderNodeCombineXYZ", -100, -400)
        tg.inputs[0].default_value = 1.0
        nt.links.new(tg.outputs[0], b.inputs["Tangent"])
        wear_col = (0.9, 0.9, 0.9, 1)
    elif finish == "BRASS":
        base = (0.80, 0.58, 0.28, 1)
        r0, r_scr, r_smear = 0.24, 0.12, 0.12
        b.inputs["Metallic"].default_value = 1.0
        brush = None
        wear_col = (0.95, 0.78, 0.45, 1)
    else:  # BLACK anodized
        base = (0.011, 0.011, 0.012, 1)
        r0, r_scr, r_smear = 0.20, 0.16, 0.14
        b.inputs["Metallic"].default_value = 0.0
        b.inputs["Specular IOR Level"].default_value = 0.5
        brush = None
        wear_col = (0.55, 0.56, 0.57, 1)  # bare aluminium

    # base colour
    col = _n(nt, "ShaderNodeRGB", -800, 1300)
    col.outputs[0].default_value = base
    colv = col.outputs[0]
    if finish == "BRASS":
        tarn = _noise(nt, pos, 14.0, 4.0, 0.6, dist=0.3, x=-800, y=1100)
        tmask = _maprange(nt, tarn.outputs["Factor"], 0.45, 0.7, 0.0, 0.55, -650, 1100)
        colv = _mix(nt, tmask, colv, (0.42, 0.30, 0.16, 1), x=-500, y=1200)
    elif finish == "BLACK":
        # faint dye unevenness
        colv = _mix(nt, _math(nt, "MULTIPLY", big.outputs["Factor"], 0.5, x=-800, y=1100), colv,
                    (0.026, 0.025, 0.028, 1), x=-500, y=1200)
    colv = _mix(nt, _math(nt, "MULTIPLY", dust, 0.85, x=-500, y=-2000), colv, (0.42, 0.40, 0.37, 1), x=-300, y=1200)
    colv = _mix(nt, edgewear, colv, wear_col, x=-150, y=1200)
    nt.links.new(colv, b.inputs["Base Color"])
    if finish == "BLACK":
        _feed(nt, b.inputs["Metallic"], edgewear)

    # roughness
    rough = _math(nt, "ADD", r0, _math(nt, "MULTIPLY", bigv, 0.05, x=-1300, y=1200), x=-1150, y=1100)
    if brush is not None:
        rough = _math(nt, "ADD", rough, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", brush, 0.5, x=-1300, y=1500),
                                                0.18, x=-1150, y=1500), x=-1000, y=1300)
    rough = _math(nt, "ADD", rough, _math(nt, "MULTIPLY", scratches, r_scr, x=-850, y=600), x=-700, y=900)
    rough = _math(nt, "ADD", rough, _math(nt, "MULTIPLY", smear, r_smear, x=-700, y=-900), x=-550, y=800)
    rough = _math(nt, "ADD", rough, _math(nt, "MULTIPLY", dust, 0.5, x=-550, y=-2000), x=-400, y=700, clamp=True)
    nt.links.new(rough, b.inputs["Roughness"])

    # normal: scratches cut in, fingerprint ridges, powder/orange-peel micro relief
    op = _noise(nt, pos, 2200.0, 2.0, 0.5, x=-800, y=-200)
    h = _math(nt, "ADD", _math(nt, "MULTIPLY", op.outputs["Factor"], 0.15, x=-650, y=-200),
              _math(nt, "MULTIPLY", scratches, -0.6, x=-650, y=-350), x=-500, y=-250)
    h = _math(nt, "ADD", h, _math(nt, "MULTIPLY", fp_total, 0.25, x=-650, y=-500), x=-350, y=-300)
    if brush is not None:
        h = _math(nt, "ADD", h, _math(nt, "MULTIPLY", brush, 0.4, x=-500, y=-450), x=-250, y=-350)
    bump = _n(nt, "ShaderNodeBump", -100, -250)
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.00002
    nt.links.new(h, bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    if finish == "BLACK":
        # thin hard sealing coat with faint smear
        b.inputs["Coat Weight"].default_value = 0.0
    o = _n(nt, "ShaderNodeOutputMaterial", 500, 200)
    nt.links.new(b.outputs[0], o.inputs["Surface"])
    return m


# ---------------------------------------------------------------------------
# 5. studio
# ---------------------------------------------------------------------------

def _obj(name, data, collection):
    o = bpy.data.objects.new(name, data)
    collection.objects.link(o)
    return o


def _mesh(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def _box_mesh(name, sx, sy, sz, z0=0.0):
    x, y = sx / 2, sy / 2
    v = [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0),
         (-x, -y, z0 + sz), (x, -y, z0 + sz), (x, y, z0 + sz), (-x, y, z0 + sz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return _mesh(name, v, f)


def _cyl_mesh(name, r, h, n=48, z0=0.0, r_top=None):
    r_top = r if r_top is None else r_top
    v, f = [], []
    for i in range(n):
        a = 2 * math.pi * i / n
        v.append((r * math.cos(a), r * math.sin(a), z0))
    for i in range(n):
        a = 2 * math.pi * i / n
        v.append((r_top * math.cos(a), r_top * math.sin(a), z0 + h))
    for i in range(n):
        j = (i + 1) % n
        f.append((i, j, n + j, n + i))
    f.append(tuple(reversed(range(n))))
    f.append(tuple(range(n, 2 * n)))
    return _mesh(name, v, f)


def _bevel(o, width, segs=3, limit="ANGLE"):
    md = o.modifiers.new("CF_Bevel", "BEVEL")
    md.width = width
    md.segments = segs
    md.limit_method = limit
    md.harden_normals = False
    return md


def _simple_mat(name, color, rough, metallic=0.0, spec=0.5, coat=0.0, bump_scale=None, bump_str=0.0,
                sheen=0.0):
    m = _new_material(name)
    nt = m.node_tree
    b = _n(nt, "ShaderNodeBsdfPrincipled", 0, 0)
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Sheen Weight"].default_value = sheen
    if bump_scale:
        tc = _n(nt, "ShaderNodeTexCoord", -800, -200)
        nz = _noise(nt, tc.outputs["Object"], bump_scale, 4.0, 0.6, x=-600, y=-200)
        bp = _n(nt, "ShaderNodeBump", -300, -200)
        bp.inputs["Strength"].default_value = bump_str
        bp.inputs["Distance"].default_value = 0.0002
        nt.links.new(nz.outputs["Factor"], bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        rr = _math(nt, "ADD", rough, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", nz.outputs["Factor"], 0.5, x=-450, y=0),
                                            0.15, x=-300, y=0), x=-150, y=100, clamp=True)
        nt.links.new(rr, b.inputs["Roughness"])
    o = _n(nt, "ShaderNodeOutputMaterial", 300, 0)
    nt.links.new(b.outputs[0], o.inputs["Surface"])
    return m


def _backdrop_material():
    """Seamless matte paper, charcoal grey with paper fibre + faint unevenness."""
    m = _new_material(PREFIX + "Backdrop")
    nt = m.node_tree
    tc = _n(nt, "ShaderNodeTexCoord", -1200, 0)
    p = tc.outputs["Object"]
    big = _noise(nt, p, 1.5, 3.0, 0.5, x=-900, y=200)
    fib = _noise(nt, p, 900.0, 3.0, 0.7, x=-900, y=-100)
    v = _maprange(nt, big.outputs["Factor"], 0.3, 0.7, 0.92, 1.06, -700, 200)
    col = _mix(nt, 1.0, (0.028, 0.028, 0.029, 1), v, "RGBA", "MULTIPLY", -500, 200)
    col = _mix(nt, _math(nt, "MULTIPLY", fib.outputs["Factor"], 0.15, x=-700, y=-100), col,
               (0.036, 0.035, 0.034, 1), x=-350, y=150)
    b = _n(nt, "ShaderNodeBsdfPrincipled", 0, 0)
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    b.inputs["Specular IOR Level"].default_value = 0.25
    b.inputs["Sheen Weight"].default_value = 0.0
    bp = _n(nt, "ShaderNodeBump", -300, -200)
    bp.inputs["Strength"].default_value = 0.15
    bp.inputs["Distance"].default_value = 0.0002
    nt.links.new(fib.outputs["Factor"], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    o = _n(nt, "ShaderNodeOutputMaterial", 300, 0)
    nt.links.new(b.outputs[0], o.inputs["Surface"])
    return m


def _sweep_mesh(name, width=3.0, front=-1.6, flat_to=0.55, radius=0.6, top=1.6, nseg=24):
    """Seamless paper sweep: floor (z=0) from y=front to y=flat_to, curving up into a back wall."""
    prof = [(front, 0.0)]
    steps = 12
    for i in range(1, steps + 1):
        prof.append((front + (flat_to - front) * i / steps, 0.0))
    for i in range(1, nseg + 1):
        a = (math.pi / 2) * i / nseg
        prof.append((flat_to + radius * math.sin(a), radius * (1 - math.cos(a))))
    prof.append((flat_to + radius, top))
    xs = [-width / 2 + width * i / 8 for i in range(9)]
    verts = []
    for x in xs:
        for (y, z) in prof:
            verts.append((x, y, z))
    n = len(prof)
    faces = []
    for i in range(len(xs) - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.append((a, a + 1, a + n + 1, a + n))
    me = _mesh(name, verts, faces)
    for p in me.polygons:
        p.use_smooth = True
    return me


def _area(name, collection, loc, target, power, size, size_y=None, shape="RECTANGLE", spread=180.0,
          temp=None, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ld.spread = math.radians(spread)
    ld.color = color
    if temp is not None and hasattr(ld, "use_temperature"):
        ld.use_temperature = True
        ld.temperature = temp
    o = _obj(name, ld, collection)
    o.location = loc
    _aim(o, target)
    return o


def _aim(o, target):
    from mathutils import Vector
    d = Vector(target) - Vector(o.location)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def _reflection_card(collection, loc, target, sx, sy, strength):
    """Emissive card seen only by glossy rays: gives the black plate a soft specular
    gradient (the classic product-shot 'strip in the reflection') without lighting the set."""
    hx, hy = sx / 2, sy / 2
    me = _mesh(PREFIX + "ReflectionCardMesh", [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)],
               [(0, 1, 2, 3)])
    o = _obj(PREFIX + "ReflectionCard", me, collection)
    o.location = loc
    _aim(o, target)
    m = _new_material(PREFIX + "ReflectionCard")
    nt = m.node_tree
    tc = _n(nt, "ShaderNodeTexCoord", -800, 0)
    sep = _n(nt, "ShaderNodeSeparateXYZ", -600, 0)
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    # soft falloff on all borders, brighter towards the top edge
    gx = _math(nt, "SINE", _math(nt, "MULTIPLY", sep.outputs[0], math.pi, x=-450, y=100), x=-300, y=100)
    gy = _maprange(nt, sep.outputs[1], 0.0, 1.0, 0.15, 1.0, -450, -100, interp="SMOOTHSTEP")
    gy2 = _math(nt, "SINE", _math(nt, "MULTIPLY", sep.outputs[1], math.pi, x=-450, y=-250), x=-300, y=-250)
    f = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", gx, gy, x=-150, y=0), gy2, x=0, y=0)
    f = _math(nt, "POWER", f, 0.7, x=150, y=0)
    em = _n(nt, "ShaderNodeEmission", 300, 0)
    em.inputs["Color"].default_value = (1.0, 0.98, 0.96, 1)
    _feed(nt, em.inputs["Strength"], _math(nt, "MULTIPLY", f, strength, x=300, y=-150))
    out = _n(nt, "ShaderNodeOutputMaterial", 500, 0)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    o.visible_camera = False
    o.visible_diffuse = False
    o.visible_shadow = False
    o.visible_transmission = False
    o.visible_volume_scatter = False
    o.visible_glossy = True
    return o


def _world(scene):
    w = bpy.data.worlds.get(PREFIX + "World") or bpy.data.worlds.new(PREFIX + "World")
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    _clear(nt)
    # dim studio: soft overhead gradient, very dark below -> natural falloff in reflections
    tc = _n(nt, "ShaderNodeTexCoord", -900, 0)
    sep = _n(nt, "ShaderNodeSeparateXYZ", -700, 0)
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    grad = _maprange(nt, sep.outputs[2], -0.2, 0.8, 0.0, 1.0, -500, 0, interp="SMOOTHSTEP")
    col = _mix(nt, grad, (0.004, 0.004, 0.0045, 1), (0.03, 0.03, 0.032, 1), x=-300, y=0)
    bg = _n(nt, "ShaderNodeBackground", 0, 0)
    nt.links.new(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    o = _n(nt, "ShaderNodeOutputWorld", 200, 0)
    nt.links.new(bg.outputs[0], o.inputs["Surface"])
    scene.world = w
    return w


def _render_settings(scene):
    scene.render.engine = "CYCLES"
    c = scene.cycles
    c.samples = 128
    c.preview_samples = 32
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.015
    c.use_denoising = True
    try:
        c.denoiser = "OPENIMAGEDENOISE"
        c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        c.denoising_prefilter = "ACCURATE"
    except Exception:
        pass
    c.max_bounces = 12
    c.diffuse_bounces = 3
    c.glossy_bounces = 4
    c.transmission_bounces = 8
    c.transparent_max_bounces = 8
    c.volume_bounces = 0
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 4.0
    c.blur_glossy = 0.5
    c.caustics_reflective = False
    c.caustics_refractive = False
    c.use_light_tree = True
    try:
        c.pixel_filter_type = "BLACKMAN_HARRIS"
        c.filter_width = 1.5
    except Exception:
        pass
    r = scene.render
    r.resolution_x, r.resolution_y = 1920, 1080
    r.resolution_percentage = 100
    r.film_transparent = False
    vs = scene.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    try:
        scene.sequencer_colorspace_settings.name = "sRGB"
    except Exception:
        pass


def _compositor(scene, vignette=0.22, grain=0.035, dispersion=0.004):
    """Photographic finishing: faint lateral CA, lens vignette, fine luminance grain.
    Best effort -- silently skipped if the compositor API differs."""
    try:
        name = PREFIX + "PhotoFinish"
        t = bpy.data.node_groups.get(name)
        if t is None:
            t = bpy.data.node_groups.new(name, "CompositorNodeTree")
        _clear(t)
        t.interface.clear()
        t.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = _n(t, "CompositorNodeRLayers", -900, 0)
        ld = _n(t, "CompositorNodeLensdist", -650, 0)
        t.links.new(rl.outputs["Image"], ld.inputs["Image"])
        ld.inputs["Dispersion"].default_value = dispersion
        img = ld.outputs["Image"]
        ic = _n(t, "CompositorNodeImageCoordinates", -650, -300)
        t.links.new(img, ic.inputs["Image"])
        # vignette: cos^4-like falloff from frame centre
        r = _vmath(t, "LENGTH", ic.outputs["Normalized"], None, -450, -300)
        vig = _maprange(t, r, 0.45, 1.45, 1.0, 1.0 - vignette, -300, -300, interp="SMOOTHSTEP")
        # grain: per-pixel noise, multiplicative (stronger in mid/low tones like film/sensor)
        nz = _noise(t, ic.outputs["Pixel"], 0.9, 0.0, 0.5, x=-450, y=-550)
        g = _math(t, "MULTIPLY", _math(t, "SUBTRACT", nz.outputs["Factor"], 0.5, x=-300, y=-550), grain * 2.0,
                  x=-150, y=-550)
        f = _math(t, "MULTIPLY", vig, _math(t, "ADD", g, 1.0, x=0, y=-550), x=150, y=-400)
        mx = _mix(t, 1.0, img, f, "RGBA", "MULTIPLY", 300, 0, clamp=False)
        go = _n(t, "NodeGroupOutput", 500, 0)
        t.links.new(mx, go.inputs[0])
        scene.compositing_node_group = t
        scene.render.use_compositing = True
        # GPU compositor needs an EGL/GL context; headless/background Blender crashes
        # without one. These 3 nodes are trivial on CPU.
        try:
            scene.render.compositor_device = "CPU"
        except Exception:
            pass
        return t
    except Exception as e:  # pragma: no cover
        print("cf_look: compositor finishing skipped:", e)
        return None


def _plate_mesh(name, size, thick, z_top, hole_r=0.0042):
    """Square slab, top at z_top. 8x8 quads on faces keeps bevel/shading clean."""
    h = size / 2
    me = _box_mesh(name, size, size, thick, z_top - thick)
    return me


def _cable(name, collection, pts, radius, mat):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = 16
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    o = _obj(name, cu, collection)
    cu.materials.append(mat)
    return o


def build_studio(scene, plate_size=0.30, plate_z=0.14, finish="BLACK", collection=None,
                 full=True):
    if collection is None:
        collection = scene.collection
    finish = (finish or "BLACK").upper()
    out = {}
    s = plate_size
    thick = 0.0015

    # ---- plate --------------------------------------------------------------
    pm = make_plate_material(finish, plate_size)
    plate = _obj(PREFIX + "Plate", _plate_mesh(PREFIX + "PlateMesh", s, thick, plate_z), collection)
    bv = _bevel(plate, 0.00045, 3)
    bv.harden_normals = True
    for p in plate.data.polygons:
        p.use_smooth = True
    plate.data.materials.append(pm)
    out["plate"] = plate

    # ---- centre bolt + washer + knurled nut on top ---------------------------
    metal = _simple_mat(PREFIX + "Steel", (0.62, 0.62, 0.62, 1), 0.28, metallic=1.0, bump_scale=3000.0, bump_str=0.2)
    washer = _obj(PREFIX + "Washer", _cyl_mesh(PREFIX + "WasherMesh", 0.0095, 0.0012, 64, plate_z), collection)
    _bevel(washer, 0.0003, 2)
    washer.data.materials.append(metal)
    nut = _obj(PREFIX + "Nut", _cyl_mesh(PREFIX + "NutMesh", 0.0072, 0.0055, 6, plate_z + 0.0012), collection)
    nut.rotation_euler[2] = math.radians(15)
    _bevel(nut, 0.0004, 2)
    nut.data.materials.append(metal)
    stud = _obj(PREFIX + "Stud", _cyl_mesh(PREFIX + "StudMesh", 0.0029, 0.0082, 32, plate_z + 0.0012), collection)
    _bevel(stud, 0.0004, 2)
    stud.data.materials.append(metal)
    for o in (washer, stud):
        for p in o.data.polygons:
            p.use_smooth = True
        o.data.set_sharp_from_angle(angle=math.radians(40))
    out["washer"], out["nut"], out["stud"] = washer, nut, stud

    # ---- wave driver below ---------------------------------------------------
    housing_mat = _simple_mat(PREFIX + "DriverBody", (0.025, 0.026, 0.028, 1), 0.42, bump_scale=1500.0,
                              bump_str=0.25, coat=0.0)
    trim_mat = _simple_mat(PREFIX + "DriverTrim", (0.20, 0.20, 0.21, 1), 0.35, metallic=1.0)
    rubber = _simple_mat(PREFIX + "Rubber", (0.006, 0.006, 0.006, 1), 0.55, sheen=0.05)
    hz0, hh = 0.008, max(plate_z - 0.06, 0.03)
    body = _obj(PREFIX + "DriverHousing", _cyl_mesh(PREFIX + "DriverMesh", 0.068, hh, 96, hz0), collection)
    _bevel(body, 0.006, 5)
    for p in body.data.polygons:
        p.use_smooth = True
    body.data.set_sharp_from_angle(angle=math.radians(50))
    body.data.materials.append(housing_mat)
    ring = _obj(PREFIX + "DriverCap", _cyl_mesh(PREFIX + "DriverCapMesh", 0.03, 0.004, 64, hz0 + hh), collection)
    _bevel(ring, 0.0012, 3)
    ring.data.materials.append(trim_mat)
    for p in ring.data.polygons:
        p.use_smooth = True
    ring.data.set_sharp_from_angle(angle=math.radians(50))
    shaft_z0 = hz0 + hh + 0.004
    shaft = _obj(PREFIX + "DriverShaft",
                 _cyl_mesh(PREFIX + "ShaftMesh", 0.0042, plate_z - thick - shaft_z0, 32, shaft_z0), collection)
    shaft.data.materials.append(metal)
    for p in shaft.data.polygons:
        p.use_smooth = True
    shaft.data.set_sharp_from_angle(angle=math.radians(40))
    # under-plate nut
    unut = _obj(PREFIX + "UnderNut", _cyl_mesh(PREFIX + "UnderNutMesh", 0.0065, 0.004, 6, plate_z - thick - 0.004),
                collection)
    _bevel(unut, 0.0004, 2)
    unut.data.materials.append(metal)
    feet = []
    for k in range(3):
        a = 2 * math.pi * k / 3 + 0.5
        f = _obj(PREFIX + "Foot%d" % k, _cyl_mesh(PREFIX + "FootMesh%d" % k, 0.011, hz0, 32, 0.0, 0.010),
                 collection)
        f.location = (0.05 * math.cos(a), 0.05 * math.sin(a), 0)
        f.data.materials.append(rubber)
        _bevel(f, 0.002, 3)
        for p in f.data.polygons:
            p.use_smooth = True
        feet.append(f)
    cable = _cable(PREFIX + "Cable", collection,
                   [(0.06, 0.03, 0.03), (0.11, 0.07, 0.004), (0.25, 0.20, 0.0035), (0.55, 0.30, 0.0035),
                    (0.9, 0.55, 0.0035)], 0.0035, rubber)
    out.update(driver=body, driver_cap=ring, shaft=shaft, cable=cable)

    if not full:
        # apparatus only: keep the user's own camera, lights, world, settings
        return out

    # ---- backdrop sweep ------------------------------------------------------
    bd = _obj(PREFIX + "Backdrop", _sweep_mesh(PREFIX + "BackdropMesh"), collection)
    bd.data.materials.append(_backdrop_material())
    out["backdrop"] = bd

    # ---- lights --------------------------------------------------------------
    c = (0, 0, plate_z)
    # key: large softbox, front-left, high
    key = _area(PREFIX + "Key", collection, (-0.95, -0.85, 1.05), (0, 0.02, plate_z), 9.0, 0.6, 0.45,
                spread=60.0, temp=5600)
    # rim / kicker: long strip, back-right, low -> rakes across the ridges
    rim = _area(PREFIX + "Rim", collection, (0.85, 0.80, plate_z + 0.13), (0, 0, plate_z), 11.0, 0.9, 0.10,
                spread=40.0, temp=5600)
    # fill: broad dim card near camera axis, right
    fill = _area(PREFIX + "Fill", collection, (0.9, -1.1, 0.55), c, 0.8, 1.2, 1.2, spread=180.0, temp=5800)
    # top bounce for a soft gradient reflection on the plate
    # reflector: big soft panel placed in the mirror direction of the hero camera so the
    # black plate carries a soft specular gradient (product-shot look)
    top = _area(PREFIX + "Top", collection, (-0.55, 1.10, 1.25), c, 2.0, 1.6, 0.5, spread=180.0, temp=5600)
    card = _reflection_card(collection, (-0.42, 0.80, 0.80), (0, 0, plate_z), 1.4, 0.32,
                            1.4 if finish == "BLACK" else 0.45)
    out.update(key_light=key, rim_light=rim, fill_light=fill, top_light=top, reflection_card=card)
    _world(scene)

    # ---- cameras -------------------------------------------------------------
    focus = bpy.data.objects.new(PREFIX + "Focus", None)
    collection.objects.link(focus)
    focus.location = (0.0, -0.03, plate_z)
    out["focus"] = focus

    def mk_cam(name, lens, loc, target, fstop, dof=True, focus_obj=None):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = 36.0
        cd.sensor_fit = "HORIZONTAL"
        cd.clip_start = 0.01
        cd.clip_end = 50.0
        cd.dof.use_dof = dof
        cd.dof.aperture_fstop = fstop
        cd.dof.aperture_blades = 9
        cd.dof.aperture_rotation = math.radians(10)
        o = _obj(name, cd, collection)
        o.location = loc
        _aim(o, target)
        if focus_obj is not None:
            cd.dof.focus_object = focus_obj
        else:
            from mathutils import Vector
            cd.dof.focus_distance = (Vector(target) - Vector(loc)).length
        return o

    # hero: 90 mm, ~33 deg elevation, 3/4 from front-left
    el, az, dist = math.radians(33.0), math.radians(-62.0), 1.45
    hero_loc = (dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az),
                plate_z + dist * math.sin(el))
    cam = mk_cam(PREFIX + "Camera", 90.0, hero_loc, (0.0, 0.0, plate_z - 0.035), 7.1, True, focus)
    top_cam = mk_cam(PREFIX + "CameraTop", 85.0, (0.0, 0.0, plate_z + 1.55), (0.0, 0.0, plate_z), 11.0, True)
    # macro: 100 mm macro lens close on a ridge
    mel, maz, md = math.radians(28.0), math.radians(-70.0), 0.36
    macro_t = (0.02, -0.03, plate_z)
    mloc = (macro_t[0] + md * math.cos(mel) * math.cos(maz), macro_t[1] + md * math.cos(mel) * math.sin(maz),
            plate_z + md * math.sin(mel))
    macro = mk_cam(PREFIX + "CameraMacro", 100.0, mloc, macro_t, 11.0, True)
    scene.camera = cam
    out.update(camera=cam, camera_top=top_cam, camera_macro=macro)

    _render_settings(scene)
    out["compositor"] = _compositor(scene)
    return out


# ======================================================================
#  Properties
# ======================================================================

class CymaticProps(bpy.types.PropertyGroup):
    sound_path: StringProperty(
        name="Audio File", subtype='FILE_PATH',
        description="Music track (.wav/.flac best; .mp3 at 170-210 kbps works)")
    grain_kind: EnumProperty(
        name="Medium",
        items=(('SAND', "Quartz Sand", "Fine natural quartz sand"),
               ('SALT', "Table Salt", "Translucent cubic salt crystals")),
        default='SAND')
    plate_finish: EnumProperty(
        name="Plate",
        items=(('BLACK', "Black Anodized", "Black anodized aluminium (classic demo)"),
               ('ALUMINUM', "Brushed Aluminium", "Brushed bare aluminium"),
               ('BRASS', "Brass", "Lightly worn brass")),
        default='BLACK')
    plate_size: FloatProperty(
        name="Plate Size", default=0.30, min=0.02, soft_max=2.0, unit='LENGTH',
        description="Edge length of the square plate (real demo plates: 0.24-0.40 m)")
    plate_height: FloatProperty(
        name="Plate Height", default=0.14, min=0.01, soft_max=1.0, unit='LENGTH',
        description="Height of the plate top above the table")
    grain_count: IntProperty(
        name="Grains", default=450000, min=1000, soft_max=1000000,
        description="Number of grains (450k fine grains = ~1 teaspoon of sand)")
    grain_radius: FloatProperty(
        name="Grain Radius", default=0.00022, min=0.00002, soft_max=0.003,
        precision=5, unit='LENGTH',
        description="Mean grain radius (fine sand 0.15-0.4 mm)")
    build_studio: BoolProperty(
        name="Studio (camera, lights, backdrop)", default=True,
        description="Build the photo studio, driver, cameras and Cycles settings")
    add_to_sequencer: BoolProperty(
        name="Add Audio to Sequencer + AV Sync", default=True)


# ======================================================================
#  Build operator
# ======================================================================

def _get_ident(mod, name):
    for it in mod.node_group.interface.items_tree:
        if getattr(it, "item_type", None) == 'SOCKET' and it.in_out == 'INPUT' \
                and it.name == name:
            return it.identifier
    return None


def _input_owner(mod, name):
    """(struct, prop) for drawing/setting a modifier input.

    Blender 5.2 exposes inputs as mod.properties.inputs.<id>.value; older
    builds used ID properties mod["<id>"]."""
    ident = _get_ident(mod, name)
    if ident is None:
        return None, None
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        item = getattr(props.inputs, ident, None)
        if item is not None and hasattr(item, "value"):
            return item, "value"
    return mod, f'["{ident}"]'


def _set_input(mod, name, value):
    """Set a modifier input by its interface name."""
    owner, prop = _input_owner(mod, name)
    if owner is None:
        return False
    try:
        if prop == "value":
            owner.value = value
        else:
            owner[prop[2:-2]] = value
        mod.id_data.update_tag()
        return True
    except (TypeError, ValueError, AttributeError):
        for it in mod.node_group.interface.items_tree:
            if getattr(it, "name", None) == name and hasattr(it, "default_value"):
                it.default_value = value
                return True
    return False


def find_sim_object(context):
    ob = context.active_object
    if ob and any(m.type == 'NODES' and m.name == "CymaticSim" for m in ob.modifiers):
        return ob
    for ob in context.scene.objects:
        if any(m.type == 'NODES' and m.name == "CymaticSim" for m in ob.modifiers):
            return ob
    return None


class CYMATICS_OT_build(bpy.types.Operator):
    bl_idname = "cymatics.build"
    bl_label = "Build Cymatic Plate"
    bl_description = "Build the plate, sand simulation, materials and studio"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        p = context.scene.cymatic_forge
        scene = context.scene
        if not p.sound_path:
            self.report({'ERROR'}, "Pick an audio file first")
            return {'CANCELLED'}
        path = bpy.path.abspath(p.sound_path)
        try:
            import aud
            aud.Sound(path).specs   # raises on missing/corrupt files
        except Exception as e:
            self.report({'ERROR'}, f"Could not read audio: {e}")
            return {'CANCELLED'}
        try:
            sound = bpy.data.sounds.load(path,
                                         check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Could not load audio: {e}")
            return {'CANCELLED'}
        if not hasattr(bpy.types, "GeometryNodeSampleSoundFrequencies"):
            self.report({'ERROR'}, "Needs Blender 5.2+ (Sample Sound Frequencies node)")
            return {'CANCELLED'}

        self._remove_previous(scene)
        coll = bpy.data.collections.new("Cymatic Forge")
        scene.collection.children.link(coll)

        # Real-world units, so physics and depth of field are true to scale.
        scene.unit_settings.system = 'METRIC'
        scene.unit_settings.scale_length = 1.0

        rig = build_studio(scene, p.plate_size, p.plate_height, p.plate_finish,
                           coll, full=p.build_studio)

        mesh = bpy.data.meshes.new("CymaticSand")
        obj = bpy.data.objects.new("CymaticSand", mesh)
        obj.location = (0.0, 0.0, p.plate_height)
        coll.objects.link(obj)
        for o in context.selected_objects:
            o.select_set(False)
        context.view_layer.objects.active = obj
        obj.select_set(True)

        shapes = build_grain_shapes(p.grain_kind)
        tree = build_main_tree("CymaticSim", shapes)
        mod = obj.modifiers.new("CymaticSim", 'NODES')
        mod.node_group = tree

        gmat = make_grain_material(p.grain_kind)
        fmat = make_fill_material(p.grain_kind, p.grain_radius)
        _set_input(mod, "Sound", sound)
        _set_input(mod, "Grain Material", gmat)
        _set_input(mod, "Pile Material", fmat)
        _set_input(mod, "Plate Size", p.plate_size)
        _set_input(mod, "Table Height", p.plate_height)
        _set_input(mod, "Grain Count", p.grain_count)
        _set_input(mod, "Grain Radius", p.grain_radius)
        if p.grain_kind == 'SALT':
            _set_input(mod, "Angle of Repose", 36.0)
            _set_input(mod, "Restitution", 0.35)
        # materials must be on the object too so Cycles/EEVEE see them
        mesh.materials.append(gmat)
        mesh.materials.append(fmat)

        if p.add_to_sequencer:
            self._setup_audio_playback(scene, p)
        scene.frame_set(scene.frame_start)
        self.report({'INFO'}, "Cymatic plate built - press Play (or bake) to run")
        return {'FINISHED'}

    @staticmethod
    def _remove_previous(scene):
        """Rebuilding replaces the previous plate instead of stacking copies."""
        for coll in [c for c in bpy.data.collections
                     if c.name.startswith("Cymatic Forge")]:
            for ob in list(coll.objects):
                data = ob.data
                bpy.data.objects.remove(ob, do_unlink=True)
                if data is not None and data.users == 0:
                    if isinstance(data, bpy.types.Mesh):
                        bpy.data.meshes.remove(data)
                    elif isinstance(data, bpy.types.Camera):
                        bpy.data.cameras.remove(data)
                    elif isinstance(data, bpy.types.Light):
                        bpy.data.lights.remove(data)
            bpy.data.collections.remove(coll)
        for ng in [g for g in bpy.data.node_groups
                   if (g.name.startswith(PREFIX) or g.name.startswith("CymaticSim"))
                   and g.users == 0]:
            bpy.data.node_groups.remove(ng)
        # nested groups lose their users only once their parents are gone
        for ng in [g for g in bpy.data.node_groups
                   if g.name.startswith(PREFIX) and g.users == 0]:
            bpy.data.node_groups.remove(ng)
        for mat in [m for m in bpy.data.materials
                    if m.name.startswith(PREFIX) and m.users == 0]:
            bpy.data.materials.remove(mat)

    def _setup_audio_playback(self, scene, p):
        try:
            se = scene.sequence_editor or scene.sequence_editor_create()
            strips = se.strips if hasattr(se, "strips") else se.sequences
            old = strips.get("CymaticAudio")
            if old is not None:
                strips.remove(old)
            strip = strips.new_sound("CymaticAudio", bpy.path.abspath(p.sound_path),
                                     1, scene.frame_start)
            dur = int(strip.frame_final_end - strip.frame_final_start)
            if dur > 0:
                scene.frame_end = scene.frame_start + dur - 1
        except (AttributeError, RuntimeError, TypeError) as e:
            print("Cymatic Forge: could not add audio strip:", e)
        try:
            scene.sync_mode = 'AUDIO_SYNC'
        except Exception:
            pass


class CYMATICS_OT_bake(bpy.types.Operator):
    bl_idname = "cymatics.bake"
    bl_label = "Bake Simulation"
    bl_description = ("Bake the sand simulation for the whole frame range so it "
                      "can be scrubbed and rendered")
    bl_options = {'REGISTER'}

    def execute(self, context):
        ob = find_sim_object(context)
        if ob is None:
            self.report({'ERROR'}, "No Cymatic plate in the scene")
            return {'CANCELLED'}
        for o in context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        context.view_layer.objects.active = ob
        return bpy.ops.object.simulation_nodes_cache_bake(selected=True)


# ======================================================================
#  UI
# ======================================================================

LIVE_INPUTS = ("Drive", "Bass Punch", "Ring Down", "Mode Min", "Mode Max",
               "Hop Height", "Node Attraction", "Angle of Repose", "Substeps",
               "Viewport Full Grains")


class CYMATICS_PT_panel(bpy.types.Panel):
    bl_label = "Cymatic Forge"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cymatics"

    def draw(self, context):
        p = context.scene.cymatic_forge
        col = self.layout.column()
        col.prop(p, "sound_path")

        box = col.box()
        box.label(text="Medium", icon='OUTLINER_DATA_POINTCLOUD')
        box.prop(p, "grain_kind", text="")
        box.prop(p, "grain_count")
        box.prop(p, "grain_radius")
        # mass of sand on the plate, for a feel of real weight
        rho = 2650.0 if p.grain_kind == 'SAND' else 2160.0
        grams = p.grain_count * 4.18879 * 0.82 * p.grain_radius ** 3 * rho * 1000.0
        box.label(text=f"~{grams:.1f} g of {'sand' if p.grain_kind == 'SAND' else 'salt'}",
                  icon='INFO')

        box = col.box()
        box.label(text="Plate", icon='MESH_PLANE')
        box.prop(p, "plate_finish", text="")
        box.prop(p, "plate_size")
        box.prop(p, "plate_height")

        col.prop(p, "build_studio")
        col.prop(p, "add_to_sequencer")
        col.separator()
        col.operator("cymatics.build", icon='PLAY_SOUND')

        ob = find_sim_object(context)
        if ob is not None:
            mod = ob.modifiers["CymaticSim"]
            box = col.box()
            box.label(text="Live (also in the modifier panel)", icon='MODIFIER')
            for nm in LIVE_INPUTS:
                owner, prop = _input_owner(mod, nm)
                if owner is not None:
                    box.prop(owner, prop, text=nm)
            col.operator("cymatics.bake", icon='FILE_CACHE')
        col.label(text="Play from the start frame; bake to scrub/render.",
                  icon='INFO')


classes = (CymaticProps, CYMATICS_OT_build, CYMATICS_OT_bake, CYMATICS_PT_panel)


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
