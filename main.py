"""Drone Flight Training Simulator - Python + Ursina (Panda3D).
Everything lives in this file: world, physics, HUD, cameras."""
import math
import random
from ursina import *

G = 9.8               # gravity (m/s^2)
GROUND = 0.35         # drone centre height when resting on the ground
MAX_LAND_SPEED = 4.5  # faster than this on touchdown = crash (m/s)
MAX_LAND_TILT = 15    # more tilt than this on touchdown = crash (deg)
DRAG = 0.6            # air drag
MAX_TILT = 25         # max pitch/roll (deg)

app = Ursina(title='Drone Flight Training Simulator', vsync=True)
window.exit_button.visible = False
Sky()
camera.position = Vec3(0, 4, -10)

def clampf(v, lo, hi):
    return max(lo, min(hi, v))

# ---------------------------------------------------------------- world
Entity(model='plane', scale=600, color=color.hsv(110, .45, .5),
       texture='white_cube', texture_scale=(300, 300))

PADS = [('HOME', Vec3(0, 0, 0), color.azure), ('PAD B', Vec3(70, 0, 60), color.violet)]
for name, p, c in PADS:
    Entity(model='plane', position=p + Vec3(0, .02, 0), scale=(9, 1, 9), color=c)
    Entity(model='plane', position=p + Vec3(0, .04, 0), scale=(6, 1, 6), color=color.white)
    Text(text=name, parent=scene, position=p + Vec3(0, 5, 0), scale=40, billboard=True,
         origin=(0, 0), color=color.black)

class Ring(Entity):
    """Orange practice ring made of 16 blocks. Fly through it for +100."""
    def __init__(self, pos, facing):
        super().__init__(position=pos, rotation_y=facing)
        self.parts = [Entity(parent=self, model='cube', color=color.orange, scale=.55,
                             position=(3 * math.cos(a), 3 * math.sin(a), 0))
                      for a in [i / 16 * math.tau for i in range(16)]]
        self.passed = False
    def mark_passed(self):
        self.passed = True
        for part in self.parts:
            part.color = color.lime

rings = []
for i in range(8):
    a = math.radians(i * 45)
    r = 40 + (i % 3) * 18
    rings.append(Ring(Vec3(math.sin(a) * r, 5 + (i % 4) * 3, math.cos(a) * r), math.degrees(a) + 90))

random.seed(42)
buildings = []  # (x, z, half_w, half_d, height)
while len(buildings) < 30:
    x, z = random.uniform(-160, 160), random.uniform(-160, 160)
    w, d, h = random.uniform(6, 14), random.uniform(6, 14), random.uniform(8, 40)
    if any(math.hypot(x - p.x, z - p.z) < 22 for _, p, _ in PADS):
        continue
    if any(math.hypot(x - r.x, z - r.z) < 14 for r in rings):
        continue
    if any(math.hypot(x - b[0], z - b[1]) < 18 for b in buildings):
        continue
    buildings.append((x, z, w / 2, d / 2, h))
    Entity(model='cube', position=(x, h / 2, z), scale=(w, h, d),
           color=color.hsv(random.choice([210, 30, 0, 180]), .12, random.uniform(.45, .85)),
           texture='white_cube', texture_scale=(w / 2, h / 2))

# ---------------------------------------------------------------- drone
drone = Entity(position=(0, GROUND, 0))
body = Entity(parent=drone, model='cube', color=color.light_gray, scale=(.9, .25, .9))
Entity(parent=drone, model='sphere', color=color.azure, scale=(.55, .3, .55), y=.12)
Entity(parent=drone, model='cube', color=color.lime, scale=(.15, .1, .2), position=(0, 0, .5))  # nose
for ang in (45, -45):
    Entity(parent=drone, model='cube', color=color.dark_gray, scale=(3.0, .08, .12), rotation_y=ang)

def make_wheel(pos, col, spin_dir):
    """Wheel-style rotor: dark tyre blocks, coloured spokes, hub."""
    w = Entity(parent=drone, position=pos)
    w.spin_dir = spin_dir
    for i in range(14):
        a = i / 14 * math.tau
        Entity(parent=w, model='cube', color=color.dark_gray, scale=(.14, .12, .24),
               position=(.55 * math.cos(a), 0, .55 * math.sin(a)), rotation_y=-math.degrees(a))
    for i in range(4):
        Entity(parent=w, model='cube', color=col, scale=(1.05, .04, .06), rotation_y=i * 45)
    Entity(parent=w, model='sphere', color=col, scale=.2)
    return w

wheels = [make_wheel((1.06, .12, 1.06), color.cyan, 1), make_wheel((-1.06, .12, 1.06), color.magenta, -1),
          make_wheel((1.06, .12, -1.06), color.magenta, -1), make_wheel((-1.06, .12, -1.06), color.cyan, 1)]

# Work out which rotation sign tilts the drone toward +x / +z (keeps controls correct).
drone.rotation = Vec3(10, 0, 0)
PITCH_SIGN = 1 if drone.up.z > 0 else -1
drone.rotation = Vec3(0, 0, 10)
ROLL_SIGN = 1 if drone.up.x > 0 else -1
drone.rotation = Vec3(0, 0, 0)

# ---------------------------------------------------------------- state
S = dict(throttle=0.0, pitch=0.0, roll=0.0, yaw=0.0, vel=Vec3(0, 0, 0), grounded=True,
         crashed=False, air=False, score=0, cam=0, orbit=0.0, msg='', msg_t=0.0)

def say(text, seconds=3.0):
    S['msg'], S['msg_t'] = text, seconds

def reset():
    drone.position = Vec3(0, GROUND, 0)
    drone.rotation = Vec3(0, 0, 0)
    body.color = color.light_gray
    S.update(throttle=0.0, pitch=0.0, roll=0.0, yaw=0.0, vel=Vec3(0, 0, 0),
             grounded=True, crashed=False, air=False)
    say('Drone reset', 1.5)

def crash(reason):
    S['crashed'] = True
    S['vel'] = Vec3(0, 0, 0)
    body.color = color.red
    say(f'CRASH: {reason}  -  press R to reset', 999)

def land_score():
    p = drone.position
    name, pad = min(((n, q) for n, q, _ in PADS), key=lambda t: math.hypot(p.x - t[1].x, p.z - t[1].z))
    d = math.hypot(p.x - pad.x, p.z - pad.z)
    if d < 4.5:
        pts = int(200 * (1 - d / 4.5))
        S['score'] += pts
        say(f'Landed on {name}: +{pts} (precision {d:.1f} m off centre)')
    else:
        say('Landed off the pads: no points')

# ---------------------------------------------------------------- physics
def fly(dt):
    k = held_keys
    S['throttle'] = clampf(S['throttle'] + (k['space'] - k['left shift']) * .6 * dt, 0, 1)
    tp = 0 if S['grounded'] else (k['w'] - k['s']) * MAX_TILT
    tr = 0 if S['grounded'] else (k['d'] - k['a']) * MAX_TILT
    S['pitch'] += (tp - S['pitch']) * min(1, 6 * dt)
    S['roll'] += (tr - S['roll']) * min(1, 6 * dt)
    if not S['grounded']:
        S['yaw'] += (k['e'] - k['q']) * 90 * dt
    drone.rotation = Vec3(S['pitch'] * PITCH_SIGN, S['yaw'], S['roll'] * ROLL_SIGN)

    vel, p = S['vel'], drone.position
    acc = drone.up * (S['throttle'] * 2 * G) + Vec3(0, -G, 0) - vel * DRAG
    if S['grounded'] and acc.y <= 0:
        acc = Vec3(0, 0, 0)
        vel = Vec3(0, 0, 0)
    vel = vel + acc * dt
    p = p + vel * dt

    if p.y < GROUND:
        tilt = math.degrees(math.acos(clampf(drone.up.y, -1, 1)))
        if vel.y < -.3 and (-vel.y > MAX_LAND_SPEED or tilt > MAX_LAND_TILT):
            drone.position = Vec3(p.x, GROUND, p.z)
            return crash('hard landing' if -vel.y > MAX_LAND_SPEED else 'tilted landing')
        if vel.y < -.3 and S['air']:
            S['air'] = False
            land_score()
        p.y = GROUND
        vel = Vec3(vel.x * max(0, 1 - 6 * dt), max(vel.y, 0), vel.z * max(0, 1 - 6 * dt))
        S['grounded'] = True
    else:
        S['grounded'] = p.y < GROUND + .02
    if p.y > GROUND + 2:
        S['air'] = True

    p.x, p.z = clampf(p.x, -290, 290), clampf(p.z, -290, 290)
    S['vel'] = vel
    drone.position = p

    for bx, bz, hw, hd, h in buildings:
        if abs(p.x - bx) < hw + .7 and abs(p.z - bz) < hd + .7 and p.y < h + .3:
            return crash('building collision')
    for r in rings:
        if not r.passed and distance(r.position, p) < 2.6:
            r.mark_passed()
            S['score'] += 100
            say('Ring! +100', 1.5)

# ---------------------------------------------------------------- camera + HUD
def move_camera(dt):
    S['orbit'] += dt * .25
    p, yr = drone.position, math.radians(S['yaw'])
    fwd = Vec3(math.sin(yr), 0, math.cos(yr))
    drone.visible = S['cam'] != 1
    if S['cam'] == 0:      # chase
        camera.position = lerp(camera.position, p - fwd * 9 + Vec3(0, 3.5, 0), min(1, 5 * dt))
        camera.look_at(p + Vec3(0, .5, 0))
    elif S['cam'] == 1:    # FPV
        camera.position = p + drone.forward * .7 + drone.up * .15
        camera.rotation = drone.rotation
    else:                  # high orbit
        camera.position = p + Vec3(math.sin(S['orbit']) * 35, 22, math.cos(S['orbit']) * 35)
        camera.look_at(p)

Entity(parent=camera.ui, model='quad', color=color.hsv(0, 0, 0, .55), scale=(.3, .19),
       origin=(-.5, .5), position=window.top_left)
hud = Text(origin=(-.5, .5), position=window.top_left + Vec2(.012, -.01), color=color.white, scale=1.1)
banner = Text(origin=(0, 0), position=(0, .38), scale=1.8, color=color.yellow)
help_txt = Text(origin=(.5, .5), position=window.top_right + Vec2(-.012, -.01), color=color.black, scale=.9,
                text='SPACE / L-SHIFT  throttle up / down\nW S  pitch     A D  roll\nQ E  yaw\n'
                     'R  reset     C  camera     H  hide help     ESC  quit')

def update():
    dt = min(time.dt, 1 / 30)
    if not S['crashed']:
        fly(dt)
    for w in wheels:
        w.rotation_y += S['throttle'] * 900 * dt * w.spin_dir
    move_camera(dt)
    v = S['vel']
    hud.text = (f"ALT {drone.y - GROUND:5.1f} m\nSPEED {v.length():5.1f} m/s\n"
                f"THROTTLE {S['throttle'] * 100:3.0f} %\nSCORE {S['score']}\n"
                f"CAM {['Chase', 'FPV', 'Orbit'][S['cam']]}")
    S['msg_t'] -= dt
    banner.text = S['msg'] if S['msg_t'] > 0 else ''

def input(key):
    if key == 'r':
        reset()
    elif key == 'c':
        S['cam'] = (S['cam'] + 1) % 3
    elif key == 'h':
        help_txt.enabled = not help_txt.enabled
    elif key == 'escape':
        application.quit()

say('Hold SPACE to lift off (hover is ~50% throttle)', 5)
app.run()
