import taichi as ti
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS. Lower the particle count if it crawls.")

import math
import time


# Cinematic black hole simulation built on the verified Taichi framebuffer path.
# No lensing or advanced post-processing yet: just stable particles, physics,
# color, glow, stars, galaxies, pulse, and an automatic opening camera orbit.

WIDTH = 1280
HEIGHT = 800
ASPECT = WIDTH / HEIGHT

PARTICLE_COUNT = 30_000
STAR_COUNT = 6_000
GALAXY_PARTICLES = 1_200

SOFTENING_EPSILON = 1e-3
GM = 7.0                         # Central gravity strength, like G*M in orbit equations.
INNER_RADIUS = 0.72              # Inner accretion disk edge, just outside the horizon.
OUTER_RADIUS = 3.9               # Outer disk radius.
EVENT_HORIZON_RADIUS = 0.55      # Black disk radius in world units.
DISK_HALF_THICKNESS = 0.025      # Vertical particle scatter for a thin elegant disk.
INWARD_DRIFT = 0.020             # Slow accretion drift toward the horizon.
SPIRAL_WAVE_STRENGTH = 0.035     # Gentle density-wave push that creates spiral structure.
TIME_STEP = 1.0 / 60.0

particle_pos = None
particle_vel = None
particle_col = None
particle_seed = None
star_pos = None
star_col = None
galaxy_pos = None
galaxy_col = None
framebuffer = None


def log(message):
    print(f"[sim] {message}", flush=True)


@ti.func
def hash11(x):
    s = ti.sin(x * 127.1 + 311.7) * 43758.5453
    return s - ti.floor(s)


@ti.func
def safe_len(v):
    return ti.sqrt(v.dot(v) + SOFTENING_EPSILON)


@ti.func
def disk_palette(radius, pulse, blue_shift):
    # Physically inspired thermal gradient:
    # white-blue hot inner gas -> golden/orange disk -> dim red outer edge.
    t = (radius - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS + SOFTENING_EPSILON)
    t = ti.max(0.0, ti.min(1.0, t))
    inner = ti.Vector([0.78, 0.92, 1.0])
    gold = ti.Vector([1.0, 0.68, 0.20])
    orange = ti.Vector([1.0, 0.30, 0.08])
    red = ti.Vector([0.42, 0.045, 0.025])

    color = inner
    if t < 0.33:
        u = t / 0.33
        color = inner * (1.0 - u) + gold * u
    elif t < 0.72:
        u = (t - 0.33) / 0.39
        color = gold * (1.0 - u) + orange * u
    else:
        u = (t - 0.72) / 0.28
        color = orange * (1.0 - u) + red * u

    heat = 1.15 - 0.72 * t
    color *= heat * (0.78 + 0.32 * pulse)
    color += ti.Vector([0.03, 0.06, 0.13]) * ti.max(0.0, blue_shift)
    return ti.min(color, ti.Vector([1.0, 1.0, 1.0]))


@ti.func
def rotate_y(p, angle):
    c = ti.cos(angle)
    s = ti.sin(angle)
    return ti.Vector([p.x * c - p.z * s, p.y, p.x * s + p.z * c])


@ti.func
def project_point(p, camera_angle, camera_pitch, camera_distance, zoom):
    rotated = rotate_y(p, camera_angle)
    cp = ti.cos(camera_pitch)
    sp = ti.sin(camera_pitch)
    tilted = ti.Vector([rotated.x, rotated.y * cp - rotated.z * sp, rotated.y * sp + rotated.z * cp])
    depth = camera_distance + tilted.z
    perspective = zoom / (depth + SOFTENING_EPSILON)
    return ti.Vector([
        0.5 + tilted.x * perspective / ASPECT,
        0.51 + tilted.y * perspective,
    ]), depth


@ti.kernel
def init_disk_particles():
    for i in range(PARTICLE_COUNT):
        idx = ti.cast(i, ti.f32)
        u = hash11(idx * 13.17 + 0.1)
        v = hash11(idx * 41.93 + 4.0)
        w = hash11(idx * 91.31 + 9.0)

        # Three arm families bias the disk into visible graceful spiral lanes.
        arm = ti.cast(i % 3, ti.f32)
        radius = INNER_RADIUS + (OUTER_RADIUS - INNER_RADIUS) * ti.sqrt(v)
        angle = 2.0 * math.pi * u + arm * 2.094 + 0.72 * radius
        # Five distinct stratified layers with randomized micro-thickness
        layer = ti.cast(i % 5, ti.f32) - 2.0
        layer_offset = layer * 0.045
        height = layer_offset + (w - 0.5) * 0.02
        p = ti.Vector([ti.cos(angle) * radius, height, ti.sin(angle) * radius])

        tangent = ti.Vector([-ti.sin(angle), 0.0, ti.cos(angle)])
        radial = ti.Vector([ti.cos(angle), 0.0, ti.sin(angle)])
        orbital_speed = ti.sqrt(GM / (radius + SOFTENING_EPSILON))

        particle_pos[i] = p
        particle_vel[i] = tangent * orbital_speed - radial * INWARD_DRIFT
        particle_seed[i] = hash11(idx * 5.113 + 22.0)
        particle_col[i] = disk_palette(radius, 1.0, 0.0)


@ti.kernel
def init_background():
    for i in range(STAR_COUNT):
        idx = ti.cast(i, ti.f32)
        x = hash11(idx * 17.1 + 1.0) * 2.0 - 1.0
        y = hash11(idx * 29.7 + 2.0) * 2.0 - 1.0
        brightness = hash11(idx * 43.3 + 3.0)
        star_pos[i] = ti.Vector([x, y])
        cool = hash11(idx * 57.9 + 4.0)
        base = ti.Vector([1.0, 0.86 + 0.14 * cool, 0.70 + 0.30 * cool])
        if brightness > 0.90:
            # Blue-giant/white star
            base = ti.Vector([0.55, 0.75, 1.0])
        elif brightness < 0.25:
            # Dim reddish/orange dwarf star
            base = ti.Vector([1.0, 0.45 + 0.15 * cool, 0.32])
        elif brightness > 0.75:
            # Warm yellow star
            base = ti.Vector([1.0, 0.95, 0.82])
        # Higher exponent creates a more realistic star brightness distribution (more dim, fewer extremely bright stars)
        star_col[i] = base * (0.05 + 0.95 * ti.pow(brightness, 3.0))

    for i in range(GALAXY_PARTICLES):
        idx = ti.cast(i, ti.f32)
        g = i % 3
        local = idx / ti.cast(GALAXY_PARTICLES, ti.f32)
        angle = 2.0 * math.pi * hash11(idx * 11.0 + 7.0)
        radius = ti.sqrt(hash11(idx * 13.0 + 8.0))
        squash = 0.18 + 0.08 * ti.cast(g, ti.f32)
        cx = -0.62 + 0.58 * ti.cast(g, ti.f32)
        cy = 0.54 - 0.21 * ti.cast(g % 2, ti.f32)
        gx = cx + ti.cos(angle) * radius * 0.12
        gy = cy + ti.sin(angle) * radius * squash * 0.12
        galaxy_pos[i] = ti.Vector([gx, gy])
        galaxy_col[i] = ti.Vector([0.28, 0.42, 0.72]) * (0.06 + 0.18 * (1.0 - radius))


@ti.kernel
def update_particles(dt: ti.f32, time_s: ti.f32):
    for i in range(PARTICLE_COUNT):
        p = particle_pos[i]
        v = particle_vel[i]
        r = safe_len(p)
        radial = p / r

        # Newtonian gravity: a = -GM * r / |r|^3, softened to avoid singularities.
        acceleration = -GM * p / ((r * r + SOFTENING_EPSILON) * r)

        angle = ti.atan2(p.z, p.x)
        arm_wave = ti.sin(3.0 * angle - 2.7 * r + time_s * 0.95 + particle_seed[i] * 0.7)
        pulse = 0.5 + 0.5 * ti.sin(time_s * 1.18 + 0.45 * ti.sin(time_s * 0.31))
        tangent = ti.Vector([-radial.z, 0.0, radial.x])

        # Breathing spiral density wave: small enough to stay graceful.
        acceleration += radial * (SPIRAL_WAVE_STRENGTH * arm_wave * (0.35 + 0.65 * pulse))
        
        layer = ti.cast(i % 5, ti.f32) - 2.0
        layer_offset = layer * 0.045
        # Restoring force to the designated layer height instead of y = 0
        acceleration += ti.Vector([0.0, -2.2 * (p.y - layer_offset), 0.0])
        
        # Gentle turbulence so the disk appears alive
        seed = particle_seed[i] * 6.28
        turb_y = ti.sin(time_s * 2.4 + p.x * 3.0 + seed) * 0.045
        turb_radial = ti.cos(time_s * 1.92 + p.z * 3.0 + seed) * 0.035
        acceleration += ti.Vector([0.0, turb_y, 0.0])
        acceleration += radial * turb_radial

        acceleration -= radial * INWARD_DRIFT * (0.55 + 0.9 / (r + 0.2))
        acceleration += tangent * 0.012 * ti.sin(time_s * 1.7 + particle_seed[i] * 6.28)

        # Semi-implicit Euler: velocity first, then position.
        v += acceleration * dt
        p += v * dt

        rr = safe_len(p)
        if rr < EVENT_HORIZON_RADIUS * 1.03 or rr > OUTER_RADIUS * 1.45 or ti.abs(p.y - layer_offset) > 0.65:
            idx = ti.cast(i, ti.f32)
            u = hash11(idx * 17.17 + time_s * 0.13)
            h = hash11(idx * 23.11 + time_s * 0.17)
            radius = OUTER_RADIUS * (0.78 + 0.20 * hash11(idx * 31.3 + time_s * 0.07))
            angle2 = 2.0 * math.pi * u + 0.62 * radius
            p = ti.Vector([ti.cos(angle2) * radius, layer_offset + (h - 0.5) * 0.02, ti.sin(angle2) * radius])
            tangent2 = ti.Vector([-ti.sin(angle2), 0.0, ti.cos(angle2)])
            radial2 = ti.Vector([ti.cos(angle2), 0.0, ti.sin(angle2)])
            v = tangent2 * ti.sqrt(GM / (radius + SOFTENING_EPSILON)) - radial2 * INWARD_DRIFT
            rr = radius

        blue_shift = -v.z * 0.055
        particle_pos[i] = p
        particle_vel[i] = v
        particle_col[i] = disk_palette(rr, pulse, blue_shift)


@ti.kernel
def clear_framebuffer():
    for x, y in ti.ndrange(WIDTH, HEIGHT):
        nx = ti.cast(x, ti.f32) / ti.cast(WIDTH, ti.f32)
        ny = ti.cast(y, ti.f32) / ti.cast(HEIGHT, ti.f32)
        
        # Nebula 1: Teal/Blue in top-left
        d1 = ti.sqrt((nx - 0.25) * (nx - 0.25) + (ny - 0.75) * (ny - 0.75) * 1.4)
        neb1 = ti.exp(-d1 * 2.5) * ti.Vector([0.005, 0.024, 0.035])
        
        # Nebula 2: Purple/Magenta in bottom-right
        d2 = ti.sqrt((nx - 0.78) * (nx - 0.78) * 1.3 + (ny - 0.28) * (ny - 0.28))
        neb2 = ti.exp(-d2 * 2.8) * ti.Vector([0.026, 0.008, 0.028])
        
        # Base cosmic dust layer (organic variations across the screen)
        dust_field = (ti.sin(nx * 4.5 + ny * 2.1) * ti.cos(ny * 3.8 - nx * 1.5) * 0.5 + 0.5) * 0.0045
        # Very fine subtle cosmic dust grain
        grain = (ti.sin(nx * 450.0 + ny * 120.0) * ti.cos(ny * 380.0 - nx * 90.0)) * 0.0012
        dust = ti.Vector([1.1, 0.9, 0.7]) * (dust_field + grain)
        
        framebuffer[x, y] = ti.Vector([0.0015, 0.0018, 0.0045]) + neb1 + neb2 + dust


@ti.kernel
def draw_background(time_s: ti.f32):
    for i in range(STAR_COUNT):
        sx = ti.cast((star_pos[i].x * 0.5 + 0.5) * WIDTH, ti.i32)
        sy = ti.cast((star_pos[i].y * 0.5 + 0.5) * HEIGHT, ti.i32)
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            twinkle = 0.76 + 0.24 * ti.sin(time_s * 0.7 + ti.cast(i, ti.f32) * 0.37)
            framebuffer[sx, sy] = ti.min(framebuffer[sx, sy] + star_col[i] * twinkle, ti.Vector([1.0, 1.0, 1.0]))

    for i in range(GALAXY_PARTICLES):
        sx = ti.cast((galaxy_pos[i].x * 0.5 + 0.5) * WIDTH, ti.i32)
        sy = ti.cast((galaxy_pos[i].y * 0.5 + 0.5) * HEIGHT, ti.i32)
        if 1 <= sx < WIDTH - 1 and 1 <= sy < HEIGHT - 1:
            for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                framebuffer[sx + dx, sy + dy] += galaxy_col[i] * ti.exp(-d2 * 0.7)


@ti.kernel
def draw_event_horizon(camera_angle: ti.f32, camera_pitch: ti.f32, camera_distance: ti.f32,
                       zoom: ti.f32, time_s: ti.f32):
    center, depth = project_point(ti.Vector([0.0, 0.0, 0.0]), camera_angle, camera_pitch, camera_distance, zoom)
    cx = ti.cast(center.x * WIDTH, ti.i32)
    cy = ti.cast(center.y * HEIGHT, ti.i32)
    radius_px = ti.cast(EVENT_HORIZON_RADIUS * zoom / (camera_distance + SOFTENING_EPSILON) * HEIGHT, ti.i32)
    radius_px = ti.max(18, radius_px)
    glow_radius = radius_px * 5
    pulse = 0.86 + 0.14 * ti.sin(time_s * 1.18)

    for dx, dy in ti.ndrange((-220, 221), (-220, 221)):
        px = cx + dx
        py = cy + dy
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            d = ti.sqrt(ti.cast(dx * dx + dy * dy, ti.f32) + SOFTENING_EPSILON)
            if d < ti.cast(glow_radius, ti.f32):
                # Brighter, more defined photon ring slightly outside horizon
                ring = ti.exp(-ti.abs(d - ti.cast(radius_px, ti.f32) * 1.25) * 0.07)
                # Soft cinematic halo that starts at horizon edge and decays smoothly outwards
                halo = ti.exp(-ti.max(0.0, d - ti.cast(radius_px, ti.f32)) * 0.012)
                glow = ti.Vector([1.0, 0.55, 0.18]) * (0.34 * ring + 0.09 * halo) * pulse
                framebuffer[px, py] = ti.min(framebuffer[px, py] + glow, ti.Vector([1.0, 1.0, 1.0]))
            if d < ti.cast(radius_px, ti.f32):
                framebuffer[px, py] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_disk(camera_angle: ti.f32, camera_pitch: ti.f32, camera_distance: ti.f32,
              zoom: ti.f32, time_s: ti.f32):
    for i in range(PARTICLE_COUNT):
        screen, depth = project_point(particle_pos[i], camera_angle, camera_pitch, camera_distance, zoom)
        sx = ti.cast(screen.x * WIDTH, ti.i32)
        sy = ti.cast(screen.y * HEIGHT, ti.i32)
        if 2 <= sx < WIDTH - 2 and 2 <= sy < HEIGHT - 2 and depth > 0.2:
            r_world = safe_len(particle_pos[i])
            near_inner = ti.exp(-ti.abs(r_world - INNER_RADIUS) * 2.0)
            brightness = 0.34 + 0.88 * near_inner
            for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                px = sx + dx
                py = sy + dy
                d2 = ti.cast(dx * dx + dy * dy, ti.f32)
                splat = ti.exp(-d2 * 0.42) * brightness
                current = framebuffer[px, py]
                framebuffer[px, py] = ti.min(current + particle_col[i] * splat, ti.Vector([1.0, 1.0, 1.0]))


def create_fields():
    global particle_pos, particle_vel, particle_col, particle_seed
    global star_pos, star_col, galaxy_pos, galaxy_col, framebuffer

    log("initialization step 1/6: allocating Taichi fields")
    particle_pos = ti.Vector.field(3, ti.f32, shape=PARTICLE_COUNT)
    particle_vel = ti.Vector.field(3, ti.f32, shape=PARTICLE_COUNT)
    particle_col = ti.Vector.field(3, ti.f32, shape=PARTICLE_COUNT)
    particle_seed = ti.field(ti.f32, shape=PARTICLE_COUNT)
    star_pos = ti.Vector.field(2, ti.f32, shape=STAR_COUNT)
    star_col = ti.Vector.field(3, ti.f32, shape=STAR_COUNT)
    galaxy_pos = ti.Vector.field(2, ti.f32, shape=GALAXY_PARTICLES)
    galaxy_col = ti.Vector.field(3, ti.f32, shape=GALAXY_PARTICLES)
    framebuffer = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))
    log("initialization step 1/6 complete: fields allocated")

    log(f"initialization step 2/6: creating thin accretion disk with {PARTICLE_COUNT:,} particles")
    init_disk_particles()
    ti.sync()
    log("verified step 1: accretion disk particles initialized")

    log("initialization step 3/6: assigning thermal color gradient")
    ti.sync()
    log("verified step 3: inner blue-white, middle gold/orange, outer dim red colors assigned")

    log(f"initialization step 4/6: creating deep-space background with {STAR_COUNT:,} stars and 3 faint galaxies")
    init_background()
    ti.sync()
    log("verified step 5: starfield and galaxies initialized")

    log("initialization step 5/6: framebuffer renderer remains active")
    clear_framebuffer()
    ti.sync()
    log("verified renderer: framebuffer clears successfully")

    log("initialization step 6/6 complete: ready for cinematic reveal")


def run_window():
    log("creating 1280x800 window")
    window = ti.ui.Window("Cinematic Black Hole - Stable Framebuffer Renderer", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    log("window object created; entering render loop")
    log("controls: SPACE pause, R reset, ESC quit")

    frame = 0
    last_time = time.perf_counter()
    start_time = last_time
    sim_time = 0.0
    paused = False
    fps_smooth = 60.0
    first_frame_verified = False

    while window.running:
        frame += 1
        now = time.perf_counter()
        frame_dt = min(1.0 / 30.0, now - last_time)
        last_time = now
        fps_smooth = fps_smooth * 0.96 + (1.0 / max(frame_dt, 1e-4)) * 0.04

        for event in window.get_events():
            if event.key == ti.ui.ESCAPE:
                log("escape received: closing window")
                window.running = False
            elif event.key == ti.ui.SPACE and event.type == ti.ui.PRESS:
                paused = not paused
                log(f"pause toggled: {paused}")
            elif event.key == 'r' and event.type == ti.ui.PRESS:
                log("R pressed: reinitializing accretion disk")
                init_disk_particles()
                ti.sync()

        if not paused:
            sim_time += frame_dt
            update_particles(TIME_STEP, sim_time)

        # Smooth camera reveal sequence lasting 10 seconds
        reveal_duration = 10.0
        reveal_raw = min(1.0, (now - start_time) / reveal_duration)
        # Smoothstep easing function
        reveal_eased = reveal_raw * reveal_raw * (3.0 - 2.0 * reveal_raw)

        # Gentle orbital motion, starting and ending smoothly (increased sweep)
        camera_angle = 0.25 + reveal_eased * 4.2 + sim_time * 0.05
        # Pitch starts at an inclined angle and eases down to a cinematic profile
        camera_pitch = 1.0 - 0.5 * reveal_eased
        # Starts significantly farther away (16.0) and moves inward (to 5.95) for a dramatic 3x dolly effect
        camera_distance = 16.0 - 10.05 * reveal_eased
        # Subtle field-of-view (zoom) animation
        zoom = 1.25 + 0.52 * reveal_eased

        clear_framebuffer()
        draw_background(sim_time)
        draw_disk(camera_angle, camera_pitch, camera_distance, zoom, sim_time)
        draw_event_horizon(camera_angle, camera_pitch, camera_distance, zoom, sim_time)
        canvas.set_image(framebuffer)
        window.show()

        if not first_frame_verified:
            log("verified steps 1-7: first cinematic frame displayed with disk, color, glow, background, pulse, and camera orbit")
            first_frame_verified = True
        if frame % 120 == 0:
            log(f"performance check: frame={frame}, estimated_fps={fps_smooth:.1f}, particles={PARTICLE_COUNT:,}")
            if fps_smooth < 30.0:
                log("performance warning: below 30 FPS; reduce PARTICLE_COUNT before removing visual effects")

    log("render loop exited normally")


def main():
    log(f"Taichi selected arch: {ti.cfg.arch}")
    create_fields()
    try:
        run_window()
    except Exception as exc:
        log(f"rendering failed on arch {ti.cfg.arch} with error: {type(exc).__name__}: {exc}")
        if ti.cfg.arch != ti.cpu:
            log("attempting automatic CPU fallback after rendering failure")
            ti.reset()
            ti.init(arch=ti.cpu)
            log(f"CPU fallback selected arch: {ti.cfg.arch}")
            create_fields()
            run_window()
        else:
            log("CPU rendering also failed; no further fallback is available")
            raise


if __name__ == "__main__":
    main()
