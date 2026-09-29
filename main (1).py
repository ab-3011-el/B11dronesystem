"""
Drone Flight Training Simulator  (Python + Ursina)
--------------------------------------------------
A transforming drone-car trainer: quadcopter flight physics, ground driving,
training rings, checkpoints, landing pads, scoring, crash detection and a
sequential mission list.

Code layout
  1. Constants & helpers
  2. World      - terrain, sky, sun, roads, pads, rings, buildings, trees, colliders
  3. Vehicle    - model, transform animation, flight physics, driving physics
  4. CameraRig  - chase / FPV / orbit cameras
  5. HUD        - telemetry, throttle gauge, toasts, banners, help
  6. Missions   - sequential training objectives
  7. Sim        - glue: input, scoring, crash handling, main update loop
"""
import math
import random

from ursina import *
from ursina.shaders import lit_with_shadows_shader, unlit_shader

# =============================================================================
# 1. CONSTANTS & HELPERS
# =============================================================================
G = 9.81                      # gravity (m/s^2)
WORLD = 1200                  # terrain edge length (m)
CELL = 10                     # terrain mesh cell size (m)
COLL_CELL = 40                # spatial-hash cell size for collision lookup (m)
PAD_R = 7.0                   # landing pad radius (m)
RING_R = 4.5                  # training ring radius (m)
MAX_TILT = 35                 # max commanded tilt (deg)
TRANSFORM_TIME = 2.6          # drone <-> car transformation duration (s)
GEAR_DRONE, GEAR_CAR = 0.45, 0.49   # body-centre height above ground per mode
CRASH_VS, CRASH_HS, CRASH_TILT = 4.5, 8.0, 22   # landing crash limits

PADS = {'A': Vec3(0, 0, 0), 'B': Vec3(200, 0, 170)}
ROADS = [(0, 85, 5, 85), (97.5, 170, 97.5, 5)]        # cx, cz, half-x, half-z
RINGS = [(0, 8, 30, 0), (10, 11, 65, 10), (35, 14, 95, 40), (75, 17, 110, 80),
         (115, 20, 100, 110), (150, 22, 80, 150), (175, 20, 45, 200), (160, 15, 10, 250)]
CHECKPOINTS = [(120, 14, -15), (70, 12, -20), (30, 10, -10)]
ROAD_CP = Vec3(100, 0, 170)
SUN_DIR = Vec3(1, -1.4, 0.7).normalized()

HELP_TEXT = ("CONTROLS\nSPACE     Throttle up\nL-SHIFT   Throttle down\nW / S     Pitch fwd / back\n"
             "A / D     Roll left / right\nQ / E     Yaw left / right\nT         Transform drone / car\n"
             "C         Change camera\nR         Reset vehicle\nH         Toggle help\nESC       Exit\n\n"
             "CAR MODE\nW / S     Accelerate / reverse\nA / D     Steer\nSPACE     Brake\n\n"
             "Hover is at about 50% throttle.\nMouse drag / wheel: orbit camera.")


def C(r, g, b, a=1.0):
    return Color(r, g, b, a)


def smooth(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def rect_dist(x, z, cx, cz, hx, hz):
    return math.hypot(max(abs(x - cx) - hx, 0), max(abs(z - cz) - hz, 0))


def terrain_h(x, z):
    """Rolling hills, flattened around pads and roads so they stay level."""
    base = (3.2 * math.sin(x * .011 + 1.3) * math.cos(z * .013)
            + 2.0 * math.sin(x * .027 + z * .019) + 1.2 * math.sin(z * .041 - x * .023))
    d = 1e9
    for p in PADS.values():
        d = min(d, math.hypot(x - p.x, z - p.z) - 12)
    for cx, cz, hx, hz in ROADS:
        d = min(d, rect_dist(x, z, cx, cz, hx, hz))
    return base * smooth((d - 4) / 18)


def calibrate_axes():
    """Measure which sign of rotation_x / rotation_z tilts the thrust forward / right."""
    p = Entity(rotation_x=10)
    ps = 1 if p.up.z > 0 else -1
    destroy(p)
    q = Entity(rotation_z=10)
    rs = 1 if q.up.x > 0 else -1
    destroy(q)
    return ps, rs


def U(**kw):
    """Entity with the unlit shader (glowing markers, UI, transparent effects)."""
    return Entity(shader=unlit_shader, **kw)


# =============================================================================
# 2. WORLD
# =============================================================================
class World:
    def __init__(self):
        self.grid = {}                 # spatial hash of colliders
        self.keepouts = []             # (x, z, radius) areas kept clear of clutter
        self.rings, self.checkpoints, self.pad_lights = [], [], []
        self.t = 0.0
        self.build_environment()
        self.build_terrain()
        self.build_roads()
        self.build_pads()
        self.build_course()
        self.build_buildings()
        self.build_trees()
        self.build_obstacles()

    # ---- sky, light, shadows ------------------------------------------------
    def build_environment(self):
        window.color = C(.55, .72, .92)
        try:
            Sky(shader=unlit_shader)
        except Exception:
            Sky()
        AmbientLight(color=C(.52, .56, .62, 1))
        try:
            self.sun = DirectionalLight(shadow_map_resolution=Vec2(2048, 2048))
        except TypeError:
            self.sun = DirectionalLight(shadows=True)
        self.sun.look_at(SUN_DIR)
        # Focus entity: two tiny cubes define the volume the shadow map covers.
        self.shadow_focus = Entity()
        for s in (-60, 60):
            Entity(parent=self.shadow_focus, model='cube', scale=.01, position=(s, s, s), shader=unlit_shader)
        try:
            scene.fog_color = C(.70, .82, .93)
            scene.fog_density = (150, 900)
        except Exception:
            pass

    def update_shadows(self, focus_pos):
        self.shadow_focus.position = focus_pos
        try:
            self.sun.position = focus_pos - SUN_DIR * 100
            self.sun.update_bounds(self.shadow_focus)
        except Exception:
            pass

    # ---- terrain mesh -------------------------------------------------------
    def build_terrain(self):
        n, half = WORLD // CELL, WORLD / 2
        verts, norms, cols, uvs, tris = [], [], [], [], []
        for iz in range(n + 1):
            for ix in range(n + 1):
                x, z = -half + ix * CELL, -half + iz * CELL
                verts.append(Vec3(x, terrain_h(x, z), z))
                gx = (terrain_h(x + 2, z) - terrain_h(x - 2, z)) / 4
                gz = (terrain_h(x, z + 2) - terrain_h(x, z - 2)) / 4
                norms.append(Vec3(-gx, 1, -gz).normalized())
                v = .86 + .14 * (.5 + .5 * math.sin(x * .09) * math.sin(z * .11 + 1))
                cols.append(C(v, v, v))
                uvs.append(Vec2(ix / n * 24, iz / n * 24))
        for iz in range(n):
            for ix in range(n):
                a = iz * (n + 1) + ix
                b, c, d = a + 1, a + n + 2, a + n + 1
                tris += [a, b, c, a, c, d]
        Entity(model=Mesh(vertices=verts, triangles=tris, normals=norms, colors=cols, uvs=uvs),
               color=C(.30, .52, .23), double_sided=True)

    # ---- roads and markings ---------------------------------------------------
    def build_roads(self):
        for cx, cz, hx, hz in ROADS:
            Entity(model='cube', color=C(.16, .17, .19), scale=(hx * 2, .1, hz * 2), position=(cx, .05, cz))
        dashes = Entity(color=C(.95, .95, .8))
        for z in range(8, 168, 8):
            Entity(parent=dashes, model='cube', scale=(.3, .02, 3), position=(0, .12, z))
        for x in range(8, 192, 8):
            Entity(parent=dashes, model='cube', scale=(3, .02, .3), position=(x, .12, 170))
        self.combine(dashes)
        # Beacon marking the "drive here" mission point on the road.
        U(model='circle', rotation_x=90, scale=14, position=ROAD_CP + Vec3(0, .2, 0),
          color=C(.1, 1, .4, .45), double_sided=True)
        U(model='cube', scale=(.5, 30, .5), position=ROAD_CP + Vec3(0, 15, 0), color=C(.1, 1, .4, .3))

    # ---- landing pads with letters and pulsing lights -----------------------
    def build_pads(self):
        try:
            from ursina.models.procedural.circle import Circle
            disc = Circle(40)
        except Exception:
            disc = 'circle'
        for name, p in PADS.items():
            Entity(model=disc, rotation_x=90, scale=PAD_R * 2.2, position=p + Vec3(0, .05, 0),
                   color=C(.9, .75, .1), double_sided=True)
            Entity(model=disc, rotation_x=90, scale=PAD_R * 2.0, position=p + Vec3(0, .07, 0),
                   color=C(.28, .30, .33), double_sided=True)
            bars = ([(-1.4, 0, .6, 5), (1.4, 0, .6, 5), (0, 2.2, 3, .6), (0, 0, 3, .6)] if name == 'A'
                    else [(-1.4, 0, .6, 5), (0, 2.2, 3, .6), (0, 0, 3, .6), (0, -2.2, 3, .6),
                          (1.4, 1.1, .6, 2), (1.4, -1.1, .6, 2)])
            for bx, bz, w, d in bars:
                Entity(model='cube', color=color.white, scale=(w, .04, d), position=p + Vec3(bx, .11, bz))
            for i in range(12):
                a = i / 12 * math.tau
                self.pad_lights.append(U(model='sphere', scale=.4, color=color.red,
                                         position=p + Vec3(math.cos(a) * (PAD_R + .4), .3, math.sin(a) * (PAD_R + .4))))
            self.keepouts.append((p.x, p.z, 40))

    # ---- rings and checkpoints ----------------------------------------------
    def build_course(self):
        for x, y, z, yaw in RINGS:
            root = Entity(position=(x, y, z), rotation_y=yaw)
            beads = [U(parent=root, model='sphere', scale=.9, color=color.orange,
                       position=(math.cos(i / 32 * math.tau) * RING_R, math.sin(i / 32 * math.tau) * RING_R, 0))
                     for i in range(32)]
            n = Vec3(math.sin(math.radians(yaw)), 0, math.cos(math.radians(yaw)))
            self.rings.append({'pos': Vec3(x, y, z), 'n': n, 'beads': beads, 'passed': False, 'prev': None})
            self.keepouts.append((x, z, 28))
        for x, y, z in CHECKPOINTS:
            e = U(model='sphere', scale=9, position=(x, y, z), color=C(.1, .9, .95, .28), double_sided=True)
            self.checkpoints.append({'pos': Vec3(x, y, z), 'entity': e, 'reached': False})
            self.keepouts.append((x, z, 25))
        for cx, cz, hx, hz in ROADS:      # keep clutter off the road corridors
            for k in range(int(max(hx, hz) * 2 // 20) + 1):
                if hz > hx:
                    self.keepouts.append((cx, cz - hz + k * 20, 24))
                else:
                    self.keepouts.append((cx - hx + k * 20, cz, 24))

    def free(self, x, z, extra=0):
        return all(math.hypot(x - kx, z - kz) > kr + extra for kx, kz, kr in self.keepouts)

    # ---- static clutter -----------------------------------------------------
    def combine(self, parent):
        try:
            parent.combine()
        except Exception:
            pass

    def register(self, item, x0, x1, z0, z1):
        for ix in range(math.floor((x0 - 4) / COLL_CELL), math.floor((x1 + 4) / COLL_CELL) + 1):
            for iz in range(math.floor((z0 - 4) / COLL_CELL), math.floor((z1 + 4) / COLL_CELL) + 1):
                self.grid.setdefault((ix, iz), []).append(item)

    def add_box(self, kind, x0, x1, y0, y1, z0, z1):
        self.register(('box', kind, x0, x1, y0, y1, z0, z1), x0, x1, z0, z1)

    def add_cyl(self, kind, x, z, r, y0, y1):
        self.register(('cyl', kind, x, z, r, y0, y1), x - r, x + r, z - r, z + r)

    def build_buildings(self):
        tints = [(.78, .74, .68), (.62, .68, .76), (.72, .60, .50), (.60, .70, .62), (.55, .57, .62)]
        made, tries = 0, 0
        while made < 36 and tries < 2000:
            tries += 1
            x, z = random.uniform(-300, 300), random.uniform(-300, 300)
            if not self.free(x, z, 12):
                continue
            w, d, hgt = random.uniform(12, 26), random.uniform(12, 26), random.uniform(18, 65)
            base = terrain_h(x, z) - 3
            r, g, b = random.choice(tints)
            Entity(model='cube', color=C(r, g, b), scale=(w, hgt, d), position=(x, base + hgt / 2, z),
                   texture='white_cube', texture_scale=(w / 4, hgt / 4))
            Entity(model='cube', color=C(.2, .2, .22), scale=(w * 1.02, .8, d * 1.02), position=(x, base + hgt + .4, z))
            U(model='sphere', scale=.8, color=color.red, position=(x, base + hgt + 1.4, z))
            self.add_box('building', x - w / 2, x + w / 2, base, base + hgt + .8, z - d / 2, z + d / 2)
            self.keepouts.append((x, z, max(w, d)))
            made += 1

    def build_trees(self):
        trunks, leaf_a, leaf_b = Entity(color=C(.40, .27, .15)), Entity(color=C(.18, .42, .18)), Entity(color=C(.25, .50, .20))
        made = 0
        while made < 110:
            x, z = random.uniform(-450, 450), random.uniform(-450, 450)
            if not self.free(x, z, -10):
                continue
            s, base = random.uniform(.8, 1.5), terrain_h(x, z)
            Entity(parent=trunks, model='cube', scale=(.6 * s, 4 * s, .6 * s), position=(x, base + 2 * s, z))
            Entity(parent=leaf_a if made % 2 else leaf_b, model='sphere', scale=(4 * s, 6 * s, 4 * s),
                   position=(x, base + 6 * s, z))
            self.add_cyl('tree', x, z, .5 * s, base, base + 4 * s)
            self.add_cyl('tree', x, z, 2 * s, base + 3 * s, base + 9 * s)
            made += 1
        for p in (trunks, leaf_a, leaf_b):
            self.combine(p)

    def build_obstacles(self):
        made, tries = 0, 0
        while made < 20 and tries < 1000:
            tries += 1
            x, z = random.uniform(-90, 260), random.uniform(-70, 230)
            if not self.free(x, z, -5):
                continue
            base = terrain_h(x, z)
            if made % 3 == 0:     # striped pylon
                h = random.uniform(9, 15)
                Entity(model='cube', color=C(.85, .2, .15), scale=(1.2, h, 1.2), position=(x, base + h / 2, z))
                U(model='sphere', scale=.7, color=color.red, position=(x, base + h + .4, z))
                self.add_box('obstacle', x - .6, x + .6, base, base + h, z - .6, z + .6)
            else:                 # crate
                s = random.uniform(1.6, 4)
                Entity(model='cube', color=C(.72, .5, .25), scale=s, position=(x, base + s / 2, z),
                       rotation_y=0, texture='white_cube')
                self.add_box('obstacle', x - s / 2, x + s / 2, base, base + s, z - s / 2, z + s / 2)
            self.keepouts.append((x, z, 6))
            made += 1

    # ---- queries --------------------------------------------------------------
    def collide(self, p, r):
        """Return the kind of collider a sphere (p, r) touches, or None."""
        for c in self.grid.get((math.floor(p.x / COLL_CELL), math.floor(p.z / COLL_CELL)), ()):
            if c[0] == 'box':
                _, kind, x0, x1, y0, y1, z0, z1 = c
                dx, dy, dz = p.x - min(max(p.x, x0), x1), p.y - min(max(p.y, y0), y1), p.z - min(max(p.z, z0), z1)
                if dx * dx + dy * dy + dz * dz < r * r:
                    return kind
            else:
                _, kind, cx, cz, cr, y0, y1 = c
                if y0 - r < p.y < y1 + r and (p.x - cx) ** 2 + (p.z - cz) ** 2 < (cr + r) ** 2:
                    return kind
        return None

    def pad_at(self, p):
        for name, c in PADS.items():
            if math.hypot(p.x - c.x, p.z - c.z) <= PAD_R:
                return name
        return None

    def update(self, dt):
        self.t += dt
        for i, light in enumerate(self.pad_lights):
            k = .5 + .5 * math.sin(self.t * 4 - (i % 12) * .52)
            light.color = C(1, .15 + .6 * k, .1, 1)
        for cp in self.checkpoints:
            cp['entity'].rotation_y += 20 * dt


# =============================================================================
# 3. VEHICLE
# =============================================================================
class Vehicle:
    def __init__(self, world, sim, ps, rs):
        self.world, self.sim, self.PS, self.RS = world, sim, ps, rs
        self.root = Entity()
        self.build_model()
        self.reset()

    # ---- model ----------------------------------------------------------------
    def build_model(self):
        r, dark = self.root, C(.10, .11, .13)
        Entity(parent=r, model='cube', color=C(.92, .94, .97), scale=(1.5, .38, 3.0))
        Entity(parent=r, model='cube', color=C(.10, .50, .95), scale=(1.52, .07, 2.3), y=-.1)
        Entity(parent=r, model='sphere', color=C(.08, .14, .24), scale=(1.0, .55, 1.7), position=(0, .28, .15))
        U(parent=r, model='cube', color=C(1, 1, .8), scale=(1.0, .08, .05), position=(0, .05, 1.52))
        U(parent=r, model='cube', color=C(1, .1, .1), scale=(1.0, .08, .05), position=(0, .05, -1.52))
        self.skids = []
        for sx in (-1, 1):
            self.skids.append(Entity(parent=r, model='cube', color=dark, scale=(.08, .06, 2.2), position=(sx * .55, -.42, 0)))

        self.modules = []
        for sx, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            hinge = Entity(parent=r, position=(sx * .75, .15, sz * 1.0))
            arm_len = math.hypot(.9, .6)
            arm = Entity(parent=hinge, model='cube', color=dark, scale=(.12, .09, arm_len),
                         position=(sx * .45, 0, sz * .3), rotation_y=math.degrees(math.atan2(sx * .9, sz * .6)))
            fan = Entity(parent=hinge, position=(sx * .9, 0, sz * .6))
            for k in range(10):                                   # duct ring
                a = k / 10 * math.tau
                Entity(parent=fan, model='cube', color=dark, position=(math.cos(a) * .62, 0, math.sin(a) * .62),
                       scale=(.09, .2, .4), rotation_y=-math.degrees(a))
            Entity(parent=fan, model='sphere', color=dark, scale=.16, y=.02)
            rotor = Entity(parent=fan, y=.06)
            for ang in (0, 90):
                Entity(parent=rotor, model='cube', color=C(.05, .05, .06), scale=(1.1, .02, .10), rotation_y=ang)
            disc = U(parent=fan, model='circle', rotation_x=90, scale=1.15, y=.08, color=C(.85, .9, 1, .05),
                     double_sided=True)
            self.modules.append({'sx': sx, 'sz': sz, 'hinge': hinge, 'arm': arm, 'fan': fan, 'rotor': rotor,
                                 'disc': disc, 'dir': 1 if sx * sz > 0 else -1, 'arm_len': arm_len})
        self.wheels = []
        for sx, sz in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
            w = Entity(parent=r)
            Entity(parent=w, model='sphere', color=C(.04, .04, .05), scale=(.28, .68, .68))
            for ang in (0, 60, 120):
                Entity(parent=w, model='cube', color=C(.7, .72, .76), scale=(.31, .05, .5), rotation_x=ang)
            self.wheels.append((w, sx, sz))

    def apply_morph(self):
        """Pose fans, arms, wheels and skids for the current transformation progress."""
        e = smooth(self.morph)
        for m in self.modules:
            sx, sz = m['sx'], m['sz']
            m['fan'].position = lerp(Vec3(sx * .9, 0, sz * .6), Vec3(sx * .32, .22, sz * .05), e)
            m['fan'].rotation_z = 90 * e                      # fan discs fold down to vertical
            m['fan'].scale = lerp(1.0, .8, e)
            m['arm'].position = lerp(Vec3(sx * .45, 0, sz * .3), Vec3(sx * .16, .1, sz * .03), e)
            m['arm'].scale = Vec3(.12, .09, lerp(m['arm_len'], .35, e))
        for w, sx, sz in self.wheels:                         # wheels deploy from the body
            w.position = lerp(Vec3(sx * .5, .1, sz * 1.0), Vec3(sx * .9, -.15, sz * 1.0), e)
            w.scale = max(.01, e)
        for s in self.skids:
            s.y, s.scale_y = lerp(-.42, -.1, e), max(.001, lerp(.06, .001, e))

    # ---- state --------------------------------------------------------------
    def reset(self):
        self.pos = Vec3(0, terrain_h(0, 0) + GEAR_DRONE, 0)
        self.vel = Vec3(0, 0, 0)
        self.yaw = self.pitch = self.roll = self.yaw_rate = 0.0
        self.throttle = self.thrust = self.speed = 0.0
        self.morph, self.morph_dir, self.transforming = 0.0, 0, False
        self.mode, self.state, self.crash_timer = 'DRONE', 'OK', 0.0
        self.grounded, self.air_time = True, 0.0
        self.rpm = self.spin = self.wheel_spin = 0.0
        self.apply_morph()
        self.apply_pose()

    @property
    def label(self):
        return 'TRANSFORMING' if self.transforming else self.mode + ' MODE'

    def gear(self):
        return lerp(GEAR_DRONE, GEAR_CAR, smooth(self.morph))

    def agl(self):
        return max(0.0, self.pos.y - terrain_h(self.pos.x, self.pos.z) - self.gear())

    def ground_speed(self):
        return abs(self.speed) if self.mode == 'CAR' else math.hypot(self.vel.x, self.vel.z)

    def apply_pose(self):
        self.root.position = self.pos
        self.root.rotation = Vec3(self.PS * self.pitch, self.yaw, self.RS * self.roll)

    # ---- transformation -------------------------------------------------------
    def request_transform(self):
        hud = self.sim.hud
        if self.state != 'OK' or self.transforming:
            return
        if self.mode == 'DRONE':
            if not self.grounded or self.vel.length() > 1.5:
                hud.toast('Land and stop before transforming to CAR mode', C(1, .7, .2))
                return
            self.morph_dir = 1
        else:
            if abs(self.speed) > 3:
                hud.toast('Stop the car before transforming to DRONE mode', C(1, .7, .2))
                return
            self.morph_dir = -1
        self.transforming, self.throttle = True, 0.0
        hud.toast('Transformation started...', C(.5, .9, 1))

    def update_transform(self, dt):
        self.morph = clamp(self.morph + self.morph_dir * dt / TRANSFORM_TIME, 0, 1)
        self.apply_morph()
        self.vel, self.speed = Vec3(0, 0, 0), 0.0
        k = 1 - math.exp(-4 * dt)
        self.pitch, self.roll = self.pitch * (1 - k), self.roll * (1 - k)
        self.pos.y = terrain_h(self.pos.x, self.pos.z) + self.gear()
        if (self.morph_dir > 0 and self.morph >= 1) or (self.morph_dir < 0 and self.morph <= 0):
            self.transforming = False
            self.mode = 'CAR' if self.morph >= 1 else 'DRONE'
            self.grounded = True
            self.sim.hud.toast(f'{self.mode} MODE ready', C(.4, 1, .5))

    # ---- drone flight -----------------------------------------------------------
    def fly(self, dt):
        k = held_keys
        if k['space']:
            self.throttle += .5 * dt
        if k['left shift']:
            self.throttle -= .8 * dt
        self.throttle = clamp(self.throttle, 0, 1)
        self.thrust += (self.throttle - self.thrust) * (1 - math.exp(-9 * dt))     # motor lag

        tp, tr = (k['w'] - k['s']) * MAX_TILT, (k['d'] - k['a']) * MAX_TILT
        if self.grounded and self.thrust < .3:                                     # sitting on the ground
            tp = tr = 0
        ka = 1 - math.exp(-5.5 * dt)
        self.pitch += (tp - self.pitch) * ka
        self.roll += (tr - self.roll) * ka
        self.yaw_rate += ((k['e'] - k['q']) * 95 - self.yaw_rate) * (1 - math.exp(-6 * dt))
        self.yaw = (self.yaw + self.yaw_rate * dt) % 360
        self.apply_pose()

        v = self.vel
        drag = Vec3(v.x * (abs(v.x) * .03 + .35), v.y * (abs(v.y) * .06 + .4), v.z * (abs(v.z) * .03 + .35))
        acc = self.root.up * (self.thrust * 2 * G) + Vec3(0, -G, 0) - drag          # lift along body-up
        self.vel = v + acc * dt
        self.pos = self.pos + self.vel * dt
        if not self.grounded:
            self.air_time += dt

        gh = terrain_h(self.pos.x, self.pos.z)
        clear = self.pos.y - gh - GEAR_DRONE
        if clear <= 0:
            if not self.grounded:
                self.touchdown()
            if self.state == 'OK':
                self.pos.y, self.grounded = gh + GEAR_DRONE, True
                self.vel.y = max(self.vel.y, 0)
                f = math.exp(-3 * dt)
                self.vel.x *= f
                self.vel.z *= f
        elif clear > .08:
            self.grounded = False
        kind = self.world.collide(self.pos, 1.4)
        if kind:
            self.sim.crash(f'{kind.capitalize()} collision', 75 if kind == 'building' else 50)
        p = 2 * (WORLD / 2 - 20)
        self.pos.x, self.pos.z = clamp(self.pos.x, -p / 2, p / 2), clamp(self.pos.z, -p / 2, p / 2)

    def touchdown(self):
        impact, hs = -self.vel.y, math.hypot(self.vel.x, self.vel.z)
        tilt = max(abs(self.pitch), abs(self.roll))
        if impact > CRASH_VS:
            return self.sim.crash(f'Excessive landing speed ({impact:.1f} m/s)')
        if hs > CRASH_HS:
            return self.sim.crash(f'Excessive ground speed at touchdown ({hs:.1f} m/s)')
        if tilt > CRASH_TILT:
            return self.sim.crash(f'Landed while tilted ({tilt:.0f} deg)')
        if impact > 3:
            self.sim.add_score(-50, f'Hard landing ({impact:.1f} m/s)')
        pad = self.world.pad_at(self.pos)
        if pad and self.air_time > 3:
            d = math.hypot(self.pos.x - PADS[pad].x, self.pos.z - PADS[pad].z)
            pts = round(300 * (1 - d / PAD_R)) + (100 if impact < 1.5 and hs < 1.5 else 0)
            self.sim.add_score(pts, f'Precision landing on PAD {pad} ({d:.1f} m off centre)')
        self.air_time = 0.0

    # ---- car driving --------------------------------------------------------------
    def drive(self, dt):
        k = held_keys
        acc = k['w'] - k['s']
        self.speed += acc * 10 * dt
        if not acc:
            self.speed -= math.copysign(min(abs(self.speed), 3 * dt), self.speed)      # rolling resistance
        if k['space']:
            self.speed -= math.copysign(min(abs(self.speed), 22 * dt), self.speed)     # brake
        self.speed = clamp(self.speed, -7, 22)
        steer = k['d'] - k['a']
        rate = steer * 55 * clamp(self.speed / 4, -1, 1) * (1 - .5 * abs(self.speed) / 22)
        self.yaw = (self.yaw + rate * dt) % 360
        yr = math.radians(self.yaw)
        fwd, rgt = Vec3(math.sin(yr), 0, math.cos(yr)), Vec3(math.cos(yr), 0, -math.sin(yr))
        old = Vec3(self.pos.x, self.pos.y, self.pos.z)
        self.pos = self.pos + fwd * self.speed * dt
        kind = self.world.collide(self.pos + Vec3(0, .5, 0), 1.5)
        if kind:
            if abs(self.speed) > 8 and kind != 'tree':
                self.sim.crash(f'{kind.capitalize()} collision at {abs(self.speed):.0f} m/s', 75 if kind == 'building' else 50)
            self.pos, self.speed = old, 0.0
        lim = WORLD / 2 - 20
        self.pos.x, self.pos.z = clamp(self.pos.x, -lim, lim), clamp(self.pos.z, -lim, lim)
        # follow the terrain: sample 4 points to derive pitch and roll
        p = self.pos
        hf, hb = terrain_h(p.x + fwd.x * 1.4, p.z + fwd.z * 1.4), terrain_h(p.x - fwd.x * 1.4, p.z - fwd.z * 1.4)
        hr, hl = terrain_h(p.x + rgt.x * .9, p.z + rgt.z * .9), terrain_h(p.x - rgt.x * .9, p.z - rgt.z * .9)
        ka = 1 - math.exp(-8 * dt)
        self.pitch += (-math.degrees(math.atan2(hf - hb, 2.8)) - self.pitch) * ka
        self.roll += (math.degrees(math.atan2(hl - hr, 1.8)) - self.roll) * ka
        self.pos.y += (terrain_h(p.x, p.z) + GEAR_CAR - self.pos.y) * (1 - math.exp(-15 * dt))
        self.vel = fwd * self.speed
        self.wheel_spin += self.PS * math.degrees(self.speed * dt / .34)
        self.grounded = True

    # ---- crash & animation ----------------------------------------------------
    def crashed(self, dt):
        self.crash_timer -= dt
        self.vel = Vec3(self.vel.x * (1 - dt), self.vel.y - G * dt, self.vel.z * (1 - dt))
        self.pos = self.pos + self.vel * dt
        floor = terrain_h(self.pos.x, self.pos.z) + .4
        if self.pos.y <= floor:
            self.pos.y, self.vel = floor, Vec3(0, 0, 0)
        else:
            self.roll += 200 * dt
            self.pitch += 120 * dt
        self.apply_pose()
        self.animate(dt)
        if self.crash_timer <= 0:
            self.sim.reset()

    def animate(self, dt):
        e = smooth(self.morph)
        target = (900 + 3000 * self.thrust) * (1 - e) if self.state == 'OK' else 0
        self.rpm += (target - self.rpm) * (1 - math.exp(-4 * dt))
        self.spin = (self.spin + self.rpm * dt) % 360
        blur = clamp(self.rpm / 3200, 0, 1)
        for m in self.modules:
            m['rotor'].rotation_y = self.spin * m['dir']
            m['disc'].color = C(.85, .9, 1, .05 + .5 * blur)          # rotor blur disc
        for w, _, _ in self.wheels:
            w.rotation_x = self.wheel_spin

    def update(self, dt):
        if self.state == 'CRASHED':
            return self.crashed(dt)
        if self.transforming:
            self.update_transform(dt)
        elif self.mode == 'DRONE':
            self.fly(dt)
        else:
            self.drive(dt)
        if self.state == 'OK':
            self.apply_pose()
            self.animate(dt)


# =============================================================================
# 4. CAMERA RIG
# =============================================================================
class CameraRig:
    MODES = ['Chase', 'FPV', 'Orbit']

    def __init__(self, vehicle, world):
        self.v, self.world, self.i = vehicle, world, 0
        self.orbit_yaw, self.orbit_pitch, self.orbit_dist = 200.0, 22.0, 10.0
        self.apply_mode()

    @property
    def name(self):
        return self.MODES[self.i]

    def cycle(self):
        self.i = (self.i + 1) % len(self.MODES)
        self.apply_mode()

    def zoom(self, d):
        self.orbit_dist = clamp(self.orbit_dist + d, 5, 30)

    def apply_mode(self):
        if self.name == 'FPV':
            camera.parent = self.v.root
            camera.position, camera.rotation = Vec3(0, .1, 1.7), Vec3(-12 * self.v.PS, 0, 0)
            camera.fov = 100
        else:
            camera.parent = scene
            camera.rotation, camera.fov = Vec3(0, 0, 0), 80
            self.snap()

    def goal(self):
        v, p = self.v, self.v.pos
        if self.name == 'Chase':
            yr, d = math.radians(v.yaw), 9 if v.mode == 'DRONE' else 7
            return p + Vec3(-math.sin(yr) * d, 3.2, -math.cos(yr) * d)
        oy, op = math.radians(self.orbit_yaw), math.radians(self.orbit_pitch)
        return p + Vec3(math.sin(oy) * math.cos(op), math.sin(op), math.cos(oy) * math.cos(op)) * self.orbit_dist

    def snap(self):
        if self.name != 'FPV':
            camera.position = self.goal()
            camera.look_at(self.v.pos + Vec3(0, .6, 0))

    def update(self, dt):
        if self.name == 'FPV':
            return
        if self.name == 'Orbit':
            if mouse.left:
                self.orbit_yaw -= mouse.velocity[0] * 150
                self.orbit_pitch = clamp(self.orbit_pitch + mouse.velocity[1] * 100, 3, 80)
            else:
                self.orbit_yaw += 8 * dt
        camera.position = lerp(camera.position, self.goal(), 1 - math.exp(-4 * dt))
        floor = terrain_h(camera.x, camera.z) + 1.2
        if camera.y < floor:
            camera.y = floor
        camera.look_at(self.v.pos + Vec3(0, .6, 0))


# =============================================================================
# 5. HUD
# =============================================================================
class HUD:
    def __init__(self, sim):
        self.sim, ui = sim, camera.ui
        tl, tr = window.top_left, window.top_right
        self.toasts, self.banner_t, self.refresh = [], 0.0, 0.0

        def panel(x, y, w, h):
            return Entity(parent=ui, model='quad', color=C(.02, .05, .09, .55), scale=(w, h),
                          origin=(-.5, .5), position=(x, y, .5))

        def text(s, x, y, origin=(-.5, .5), scale=1.0, col=C(.85, .95, 1)):
            return Text(s, parent=ui, position=(x, y), origin=origin, scale=scale, color=col)

        panel(tl.x + .015, tl.y - .015, .40, .25)
        self.left = text('', tl.x + .03, tl.y - .03)
        panel(tr.x - .335, tr.y - .015, .32, .15)
        self.right = text('', tr.x - .32, tr.y - .03)
        self.help_panel = panel(tr.x - .335, tr.y - .18, .32, .60)
        self.help = text(HELP_TEXT, tr.x - .32, tr.y - .195, scale=.85)
        panel(-.42, -.395, .84, .085)
        self.mission = text('', 0, -.4375, origin=(0, 0), scale=1.0, col=C(1, .9, .5))
        gx, gy, gh = tl.x + .06, -.36, .30                    # vertical throttle gauge
        Entity(parent=ui, model='quad', color=C(.05, .08, .12, .8), scale=(.036, gh), origin=(0, -.5), position=(gx, gy, .2))
        self.thr_fill = Entity(parent=ui, model='quad', color=C(.2, .8, .4), scale=(.028, .001),
                               origin=(0, -.5), position=(gx, gy, .1))
        Entity(parent=ui, model='quad', color=C(1, 1, 1, .95), scale=(.055, .003), position=(gx, gy + gh / 2, .05))
        text('THR', gx, gy - .01, origin=(0, .5), scale=.8)
        self.thr_txt = text('0%', gx, gy + gh + .012, origin=(0, -.5), scale=.9)
        self.gauge_h, self.gauge_y = gh, gy
        self.toast_txt = text('', 0, .32, origin=(0, .5), scale=1.05, col=C(1, .9, .4))
        self.banner = text('', 0, .08, origin=(0, 0), scale=3.2, col=C(1, .15, .15))
        self.banner_sub = text('', 0, -.02, origin=(0, 0), scale=1.4, col=C(1, .85, .85))

    def toast(self, msg, col=C(1, .9, .4), dur=4.0):
        self.toasts.append([msg, dur, col])
        self.toasts = self.toasts[-4:]

    def show_banner(self, title, sub, dur=3.0):
        self.banner.text, self.banner_sub.text, self.banner_t = title, sub, dur

    def clear_banner(self):
        self.banner.text = self.banner_sub.text = ''
        self.banner_t = 0

    def toggle_help(self):
        self.help.enabled = self.help_panel.enabled = not self.help.enabled

    def update(self, dt):
        s, v = self.sim, self.sim.vehicle
        self.toasts = [[m, t - dt, c] for m, t, c in self.toasts if t - dt > 0]
        if self.banner_t > 0:
            self.banner_t -= dt
            if self.banner_t <= 0:
                self.clear_banner()
        self.thr_fill.scale_y = max(.001, self.gauge_h * v.throttle)
        self.refresh -= dt
        if self.refresh > 0:
            return
        self.refresh = .1                                     # text refresh at 10 Hz
        spd = v.ground_speed()
        state = 'CRASHED' if v.state == 'CRASHED' else ('ON GROUND' if v.grounded else 'AIRBORNE')
        self.left.text = (f'{v.label}\nALTITUDE   {v.agl():7.1f} m\nSPEED      {spd:6.1f} m/s ({spd * 3.6:3.0f} km/h)\n'
                          f'V.SPEED    {v.vel.y if v.mode == "DRONE" else 0:+6.1f} m/s\n'
                          f'POSITION   {v.pos.x:6.0f} {v.pos.y:5.0f} {v.pos.z:6.0f}\nSTATUS     {state}')
        self.right.text = f'SCORE   {s.score}\nCAMERA  {s.cams.name}\nFPS     {s.fps:3.0f}'
        self.thr_txt.text = f'{v.throttle * 100:.0f}%'
        self.mission.text = s.missions.status()
        self.toast_txt.text = '\n'.join(m for m, _, _ in self.toasts)


# =============================================================================
# 6. MISSIONS
# =============================================================================
class MissionManager:
    def __init__(self, sim):
        self.sim, self.idx, self.done, self.drove = sim, 0, False, False
        w, v = sim.world, sim.vehicle

        def landed(pad):
            return (v.mode == 'DRONE' and not v.transforming and v.grounded and v.vel.length() < .6
                    and w.pad_at(v.pos) == pad and v.state == 'OK')

        def at_road_cp():
            return v.mode == 'CAR' and math.hypot(v.pos.x - ROAD_CP.x, v.pos.z - ROAD_CP.z) < 9

        def drive_done():
            if at_road_cp():
                self.drove = True
            return self.drove

        rings = lambda: sum(r['passed'] for r in w.rings)
        cps = lambda: sum(c['reached'] for c in w.checkpoints)
        # (title, completion test, progress text)
        self.steps = [
            ('Take off and climb above 10 m', lambda: v.mode == 'DRONE' and v.agl() > 10, lambda: f'{v.agl():.0f}/10 m'),
            (f'Fly through all {len(RINGS)} training rings', lambda: rings() == len(RINGS), lambda: f'{rings()}/{len(RINGS)}'),
            (f'Reach all {len(CHECKPOINTS)} checkpoints', lambda: cps() == len(CHECKPOINTS), lambda: f'{cps()}/{len(CHECKPOINTS)}'),
            ('Land on PAD A', lambda: landed('A'), lambda: ''),
            ('Transform into CAR mode (T)', lambda: v.mode == 'CAR', lambda: ''),
            ('Drive along the road to the green beacon', drive_done, lambda: ''),
            ('Transform back into DRONE mode (T)', lambda: self.drove and v.mode == 'DRONE', lambda: ''),
            ('Take off and land on PAD B', lambda: self.drove and landed('B'), lambda: ''),
        ]

    def update(self):
        if self.done:
            return
        title, test, _ = self.steps[self.idx]
        if test():
            self.sim.add_score(150, f'Mission complete: {title}')
            self.idx += 1
            if self.idx >= len(self.steps):
                self.done = True
                self.sim.add_score(500, 'ALL MISSIONS COMPLETE')
                self.sim.hud.show_banner('MISSION COMPLETE', f'Final score {self.sim.score}', 8)

    def status(self):
        if self.done:
            return 'ALL MISSIONS COMPLETE - free flight'
        title, _, prog = self.steps[self.idx]
        return f'MISSION {self.idx + 1}/{len(self.steps)}: {title}  {("[" + prog() + "]") if prog() else ""}'


# =============================================================================
# 7. SIMULATION (glue)
# =============================================================================
class Sim:
    def __init__(self):
        ps, rs = calibrate_axes()
        Entity.default_shader = lit_with_shadows_shader      # world + vehicle: lit with shadows
        self.world = World()
        self.vehicle = Vehicle(self.world, self, ps, rs)
        Entity.default_shader = unlit_shader                 # HUD / UI: flat
        self.score, self.fps, self.shadow_t = 0, 60.0, 0.0
        self.hud = HUD(self)
        self.cams = CameraRig(self.vehicle, self.world)
        self.missions = MissionManager(self)
        self.hud.toast('Welcome, pilot. Hold SPACE to spool up - hover is near 50% throttle.', dur=7)

    def add_score(self, pts, reason):
        self.score += pts
        self.hud.toast(f'{pts:+d}  {reason}', C(.4, 1, .5) if pts >= 0 else C(1, .45, .35), 5)

    def crash(self, reason, extra=0):
        v = self.vehicle
        if v.state != 'OK':
            return
        v.state, v.crash_timer, v.thrust, v.throttle = 'CRASHED', 3.0, 0.0, 0.0
        v.transforming = False
        self.add_score(-(100 + extra), f'Crash: {reason}')
        self.hud.show_banner('CRASH DETECTED', reason, 3.0)

    def reset(self):
        self.vehicle.reset()
        for r in self.world.rings:
            r['prev'] = None
        self.hud.clear_banner()
        self.cams.snap()

    def on_input(self, key):
        if key == 'escape':
            application.quit()
        elif key == 'r':
            self.reset()
        elif key == 'c':
            self.cams.cycle()
        elif key == 'h':
            self.hud.toggle_help()
        elif key == 't':
            self.vehicle.request_transform()
        elif key == 'scroll up':
            self.cams.zoom(-1)
        elif key == 'scroll down':
            self.cams.zoom(1)

    def check_rings_and_checkpoints(self):
        p = self.vehicle.pos
        for i, r in enumerate(self.world.rings):
            rel = p - r['pos']
            d = rel.dot(r['n'])
            if not r['passed'] and r['prev'] is not None and r['prev'] * d < 0 and rel.length() < 15:
                lateral = (rel - r['n'] * d).length()
                if lateral < RING_R - .6:
                    r['passed'] = True
                    for b in r['beads']:
                        b.color = color.lime
                    self.add_score(100, f'Ring {i + 1} cleared')
            r['prev'] = d
        for i, c in enumerate(self.world.checkpoints):
            if not c['reached'] and (p - c['pos']).length() < 7:
                c['reached'] = True
                c['entity'].enabled = False
                self.add_score(75, f'Checkpoint {i + 1} reached')

    def update(self, dt):
        dt = min(dt, 1 / 30)
        self.fps += (1 / max(dt, 1e-4) - self.fps) * .1
        self.vehicle.update(dt)
        if self.vehicle.state == 'OK':
            self.check_rings_and_checkpoints()
            self.missions.update()
        self.world.update(dt)
        self.cams.update(dt)
        self.hud.update(dt)
        self.shadow_t -= dt
        if self.shadow_t <= 0:
            self.shadow_t = .25
            self.world.update_shadows(self.vehicle.pos)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    random.seed(7)
    app = Ursina(title='Drone Flight Training Simulator', borderless=False)
    window.exit_button.visible = False
    window.fps_counter.enabled = False
    sim = Sim()

    def update():          # called every frame by Ursina
        sim.update(time.dt)

    def input(key):        # called on every key / mouse event by Ursina
        sim.on_input(key)

    app.run()
