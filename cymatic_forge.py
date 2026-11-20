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
                ("Smoothing", 'NodeSocketInt', 2, 0, 20)),
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
    ch = b.group(chladni, Position=P, M=I["M"], N=I["N"], Sign=I["Sign"],
                 K=b.div(math.pi, Lh))
    F = ch.outputs["F"]
    gradF = ch.outputs["Gradient"]
    gamma = b.mul(I["Drive"], b.mul(b.m('ABSOLUTE', F), 0.5))
    excess = b.mx(b.sub(gamma, 1.0), 0.0)
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

    # resting grains: sit on the pile, avalanche if slope > repose
    smag = b.v('LENGTH', slope)
    over = b.mx(b.sub(smag, I["Repose Tan"]), 0.0)
    down = b.v('SCALE', b.v('NORMALIZE', slope),
               scale=b.mul(b.mul(b.mul(over, I["Avalanche Speed"]), dt), -1.0))
    dnx, dny, _ = b.sep(down)
    # tiny settling shiver while the plate is excited but below 1 g
    shiver = b.mul(b.mul(b.clamp01(b.mul(gamma, 1.5)), dt), 0.0004)
    sj = b.rand('FLOAT_VECTOR', (-1.0, -1.0, 0.0), (1.0, 1.0, 0.0), idn,
                b.imath('ADD', I["Seed"], 303))
    sjx, sjy, _ = b.sep(sj)
    # vibration creep: below 1 g grains still slide/roll down the oscillating
    # plate tilt towards the nodal line (stalls where the plate is still)
    dist = b.div(b.m('ABSOLUTE', F), b.mx(b.v('LENGTH', gradF), 1e-6))
    creep = b.mn(b.mul(b.mul(b.clamp01(gamma), I["Creep"]), dt),
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
    ("Sand", "Grain Count", 'NodeSocketInt', 200000, 100, 5000000,
     "Number of grains sprinkled at the first frame"),
    ("Sand", "Grain Radius", 'NodeSocketFloat', 0.00035, 0.00002, 0.02,
     "Mean grain radius in metres (fine sand ~0.15-0.4 mm)"),
    ("Sand", "Size Variation", 'NodeSocketFloat', 0.4, 0.0, 0.9, ""),
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
    ("Plate", "Pile Resolution", 'NodeSocketInt', 192, 16, 1024,
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
        "Avalanche Speed": b.mul(I["Grain Radius"], 60.0)})
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
    body_z = b.mx(b.sub(hp, b.mul(I["Grain Radius"], 1.4)), -0.0006)
    pile = b.set_pos(pile, position=b.xyz(ppx, ppy, body_z))
    # trim cells that hold no sand at all
    del_n = b.node("GeometryNodeDeleteGeometry", domain='POINT')
    b.feed(del_n.inputs["Geometry"], pile)
    b.feed(del_n.inputs["Selection"], b.lt(hp, b.mul(I["Grain Radius"], 0.9)))
    pile = del_n.outputs[0]
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
#  LOOK (materials, grain shapes, studio) - placeholder, replaced later
# ======================================================================

def build_grain_shapes(kind):
    tree, b, gi, go = _new_group(PREFIX + "Grain Shapes " + kind.title(),
                                 outputs=(("Instances", 'NodeSocketGeometry'),))
    join = b.node("GeometryNodeJoinGeometry")
    for i in range(6):
        ico = b.node("GeometryNodeMeshIcoSphere")
        ico.inputs["Radius"].default_value = 1.0
        ico.inputs["Subdivisions"].default_value = 2
        gti = b.node("GeometryNodeGeometryToInstance")
        b.L.new(ico.outputs["Mesh"], gti.inputs[0])
        b.L.new(gti.outputs[0], join.inputs[0])
    b.L.new(join.outputs[0], go.inputs[0])
    _layout(tree)
    return tree


def make_grain_material(kind):
    mat = bpy.data.materials.new(PREFIX + "Grain")
    return mat


def make_fill_material(kind):
    return bpy.data.materials.new(PREFIX + "Pile")


def make_plate_material(finish):
    return bpy.data.materials.new(PREFIX + "Plate")


def build_studio(scene, plate_size, plate_z, finish, collection, full=True):
    me = bpy.data.meshes.new(PREFIX + "Plate")
    s = plate_size / 2
    t = 0.0015
    verts = [(x, y, z) for z in (plate_z - t, plate_z) for y in (-s, s) for x in (-s, s)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    me.from_pydata(verts, [], faces)
    ob = bpy.data.objects.new(PREFIX + "Plate", me)
    collection.objects.link(ob)
    me.materials.append(make_plate_material(finish))
    return {"plate": ob}


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
        name="Grains", default=200000, min=1000, soft_max=1000000,
        description="Number of grains (200k = ~1 tablespoon of fine sand)")
    grain_radius: FloatProperty(
        name="Grain Radius", default=0.00035, min=0.00002, soft_max=0.003,
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
        fmat = make_fill_material(p.grain_kind)
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
