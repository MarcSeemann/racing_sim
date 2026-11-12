extends Node3D
## carsim 3D front end.
##
## The physics runs in Python (python -m carsim.server). This scene connects to it over
## TCP, builds the track from the centreline it sends, draws the car where the
## simulation says it is, and sends the driver's inputs back.
##
## Simulation axes: x east, y north, z up. Godot axes: x right, y up, -z forward.
## A simulation point (x, y, z) is drawn at Godot (x, z, -y).

const HOST := "127.0.0.1"
const PORT := 4545

var peer := StreamPeerTCP.new()
var buf := ""
var connected := false
var track := {}
var state := {}
var have_state := false

var car_root: Node3D
var body: Node3D
var wheels := []          # [steer_pivot, spin_node] per wheel: fl, fr, rl, rr
var camera: Camera3D
var cam_mode := 0         # 0 chase, 1 cockpit, 2 TV
var cam_dir := Vector3(0, 0, -1)
const CAM_NAMES := ["chase", "cockpit", "TV"]
var tv_points := []
var sun: DirectionalLight3D
var env: Environment
var sky_mat: ProceduralSkyMaterial
var asphalt: StandardMaterial3D
var rain: CPUParticles3D
var line_mesh: MeshInstance3D
var hud := {}
var input_timer := 0.0
var retry_timer := 0.0
var prev_keys := {}

# Testing aid: CARSIM_SHOT=/path.png CARSIM_SHOT_AFTER=8 saves a screenshot after N seconds of
# driving and quits; CARSIM_CAM=0/1/2 picks the camera. Unset in normal use.
var shot_path := OS.get_environment("CARSIM_SHOT")
var shot_after := float(OS.get_environment("CARSIM_SHOT_AFTER")) if OS.get_environment("CARSIM_SHOT_AFTER") != "" else 8.0
var drive_time := 0.0

# ---- setup ----------------------------------------------------------------------
func _ready() -> void:
	if OS.get_environment("CARSIM_CAM") != "":
		cam_mode = int(OS.get_environment("CARSIM_CAM"))
	_build_environment()
	_build_hud()
	_set_status("Connecting to the physics server on %s:%d ...\nStart it with:  python -m carsim.server" % [HOST, PORT])
	peer.connect_to_host(HOST, PORT)

func sim_to_godot(x: float, y: float, z: float) -> Vector3:
	return Vector3(x, z, -y)

func _build_environment() -> void:
	var we := WorldEnvironment.new()
	env = Environment.new()
	sky_mat = ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.32, 0.52, 0.85)
	sky_mat.sky_horizon_color = Color(0.72, 0.8, 0.9)
	sky_mat.ground_horizon_color = Color(0.5, 0.55, 0.5)
	sky_mat.ground_bottom_color = Color(0.2, 0.25, 0.2)
	var sky := Sky.new()
	sky.sky_material = sky_mat
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.62, 0.64, 0.66)  # neutral, so the asphalt is not tinted blue
	env.ambient_light_energy = 0.55
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	env.fog_enabled = true
	env.fog_light_color = Color(0.75, 0.8, 0.88)
	env.fog_density = 0.0008
	we.environment = env
	add_child(we)
	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 35, 0)
	sun.light_energy = 1.2
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = 250.0
	add_child(sun)
	camera = Camera3D.new()
	camera.fov = 70
	camera.far = 3000
	add_child(camera)
	camera.position = Vector3(0, 30, 30)
	camera.look_at(Vector3.ZERO)
	rain = CPUParticles3D.new()
	rain.amount = 3000
	rain.lifetime = 0.9
	rain.emission_shape = CPUParticles3D.EMISSION_SHAPE_BOX
	rain.emission_box_extents = Vector3(30, 1, 30)
	rain.direction = Vector3(0, -1, 0)
	rain.spread = 3
	rain.gravity = Vector3(0, -30, 0)
	rain.initial_velocity_min = 25
	rain.initial_velocity_max = 30
	var drop := QuadMesh.new()
	drop.size = Vector2(0.02, 0.45)
	var dm := StandardMaterial3D.new()
	dm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	dm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	dm.albedo_color = Color(0.75, 0.82, 0.95, 0.45)
	dm.billboard_mode = BaseMaterial3D.BILLBOARD_FIXED_Y
	drop.material = dm
	rain.mesh = drop
	rain.emitting = false
	add_child(rain)

func _mat(color: Color, rough := 0.8, metal := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = rough
	m.metallic = metal
	return m

# ---- track ------------------------------------------------------------------------
func _build_track() -> void:
	var xs: Array = track["x"]
	var ys: Array = track["y"]
	var zs: Array = track["z"]
	var hs: Array = track["heading"]
	var n := xs.size()
	var half: float = track["width"] / 2.0
	var kerb: float = track["kerb"]
	# Corners: where the heading changes quickly (kerbs go there).
	var corner := []
	corner.resize(n)
	for i in n:
		var dh: float = wrapf(float(hs[(i + 4) % n]) - float(hs[(i - 4 + n) % n]), -PI, PI)
		corner[i] = abs(dh) / 8.0 > 1.0 / 350.0
	var bands := [
		# [offset from, offset to, kind]
		[-half, half, "asphalt"],
		[half, half + kerb, "kerb_l"], [-half - kerb, -half, "kerb_r"],
		[half, half + 45.0, "grass"], [-half - 45.0, -half, "grass"],
		[half - 0.35, half - 0.15, "line"], [-half + 0.15, -half + 0.35, "line"],
	]
	asphalt = _mat(Color(0.16, 0.16, 0.17), 0.95)
	asphalt.metallic_specular = 0.2
	var grass_mat := _mat(Color(0.12, 0.26, 0.1), 1.0)
	grass_mat.metallic_specular = 0.1
	var mats := {"asphalt": asphalt, "grass": grass_mat,
		"line": _mat(Color(0.92, 0.92, 0.92), 0.6)}
	var kerb_mat := StandardMaterial3D.new()
	kerb_mat.vertex_color_use_as_albedo = true
	kerb_mat.roughness = 0.6
	for band in bands:
		var st := SurfaceTool.new()
		st.begin(Mesh.PRIMITIVE_TRIANGLES)
		var kind: String = band[2]
		var lift := 0.0
		if kind == "line":
			lift = 0.01
		elif kind.begins_with("kerb"):
			lift = 0.03
		elif kind == "grass":
			lift = -0.04
		for i in n:
			var j := (i + 1) % n
			if kind.begins_with("kerb") and not (corner[i] and corner[j]):
				continue
			var a0 := _edge(xs, ys, zs, hs, i, band[0], lift)
			var a1 := _edge(xs, ys, zs, hs, i, band[1], lift)
			var b0 := _edge(xs, ys, zs, hs, j, band[0], lift)
			var b1 := _edge(xs, ys, zs, hs, j, band[1], lift)
			if kind.begins_with("kerb"):
				var c := Color(0.85, 0.1, 0.1) if (i / 3) % 2 == 0 else Color(0.95, 0.95, 0.95)
				st.set_color(c)
			# Godot front faces are clockwise seen from the front (here: from above).
			for v in [a0, a1, b0, a1, b1, b0]:
				st.add_vertex(v)
		var mi := MeshInstance3D.new()
		st.generate_normals()
		mi.mesh = st.commit()
		mi.material_override = kerb_mat if kind.begins_with("kerb") else mats[kind]
		add_child(mi)
	# Start / finish line.
	var sf := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = Vector3(track["width"], 0.02, 1.2)
	sf.mesh = bm
	sf.material_override = _mat(Color(0.95, 0.95, 0.95), 0.5)
	add_child(sf)
	sf.position = sim_to_godot(xs[0], ys[0], zs[0]) + Vector3(0, 0.02, 0)
	sf.rotation.y = float(hs[0]) - PI / 2
	# Ground far below the lowest point, so the horizon is not empty.
	var ground := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(6000, 6000)
	ground.mesh = pm
	ground.material_override = _mat(Color(0.11, 0.24, 0.09), 1.0)
	add_child(ground)
	ground.position = Vector3(0, float(zs.min()) - 1.5, 0)
	# Racing line (toggle with L).
	var lst := SurfaceTool.new()
	lst.begin(Mesh.PRIMITIVE_LINE_STRIP)
	var lx: Array = track["line_x"]
	var ly: Array = track["line_y"]
	for i in lx.size() + 1:
		var k := i % lx.size()
		lst.add_vertex(sim_to_godot(lx[k], ly[k], zs[k]) + Vector3(0, 0.05, 0))
	line_mesh = MeshInstance3D.new()
	line_mesh.mesh = lst.commit()
	var lm := _mat(Color(0.35, 0.6, 1.0))
	lm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	line_mesh.material_override = lm
	line_mesh.visible = false
	add_child(line_mesh)
	# TV camera spots: every ~250 m, set back from the outside of the track.
	for i in range(0, n, 250):
		tv_points.append(_edge(xs, ys, zs, hs, i, -(half + 25.0), 8.0))

func _edge(xs: Array, ys: Array, zs: Array, hs: Array, i: int, offset: float, lift: float) -> Vector3:
	var h: float = hs[i]
	var x: float = float(xs[i]) - offset * sin(h)
	var y: float = float(ys[i]) + offset * cos(h)
	return sim_to_godot(x, y, float(zs[i]) + lift)

# ---- car --------------------------------------------------------------------------
func _build_car() -> void:
	car_root = Node3D.new()
	add_child(car_root)
	body = Node3D.new()
	car_root.add_child(body)
	var a: float = track["a"]
	var b: float = track["b"]
	var tf: float = track["track_front"]
	var trr: float = track["track_rear"]
	var r: float = track["wheel_radius"]
	var paint := _mat(Color(0.72, 0.05, 0.04), 0.35, 0.3)
	var dark := _mat(Color(0.08, 0.08, 0.09), 0.5)
	# Forward is -Z. Parts: [size, position, material]
	var parts := [
		[Vector3(0.55, 0.38, 2.6), Vector3(0, 0.42, -0.6), paint],          # nose and monocoque
		[Vector3(0.3, 0.22, 1.2), Vector3(0, 0.33, -a - 0.2), paint],        # nose tip
		[Vector3(1.6, 0.38, 1.8), Vector3(0, 0.36, 0.5), paint],             # sidepods
		[Vector3(0.7, 0.55, 1.6), Vector3(0, 0.62, 1.0), paint],             # engine cover
		[Vector3(1.9, 0.05, 0.55), Vector3(0, 0.14, -a - 0.55), dark],       # front wing
		[Vector3(1.0, 0.06, 0.35), Vector3(0, 1.0, b + 0.45), dark],         # rear wing
		[Vector3(0.05, 0.5, 0.4), Vector3(0.5, 0.78, b + 0.45), dark],       # rear wing endplates
		[Vector3(0.05, 0.5, 0.4), Vector3(-0.5, 0.78, b + 0.45), dark],
		[Vector3(1.3, 0.04, 3.6), Vector3(0, 0.08, 0.1), dark],              # floor
	]
	for p in parts:
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = p[0]
		mi.mesh = bm
		mi.material_override = p[2]
		mi.position = p[1]
		body.add_child(mi)
	var helmet := MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.15
	sm.height = 0.3
	helmet.mesh = sm
	helmet.material_override = _mat(Color(0.95, 0.85, 0.1), 0.3)
	helmet.position = Vector3(0, 0.78, -0.1)
	body.add_child(helmet)
	# Wheels: a steer pivot (front only turns) holding a spinning tyre with a visible spoke.
	var tyre_mat := _mat(Color(0.06, 0.06, 0.06), 0.9)
	var rim_mat := _mat(Color(0.75, 0.75, 0.78), 0.3, 0.8)
	var spots := [Vector3(-tf / 2, r, -a), Vector3(tf / 2, r, -a), Vector3(-trr / 2, r, b), Vector3(trr / 2, r, b)]
	for k in 4:
		var pivot := Node3D.new()
		pivot.position = spots[k]
		car_root.add_child(pivot)
		var spin := Node3D.new()
		pivot.add_child(spin)
		var tyre := MeshInstance3D.new()
		var cm := CylinderMesh.new()
		cm.top_radius = r
		cm.bottom_radius = r
		cm.height = 0.38 if k >= 2 else 0.32
		tyre.mesh = cm
		tyre.rotation.z = PI / 2
		tyre.material_override = tyre_mat
		spin.add_child(tyre)
		var spoke := MeshInstance3D.new()
		var sbm := BoxMesh.new()
		sbm.size = Vector3(0.4, 0.08, r * 1.5)
		spoke.mesh = sbm
		spoke.material_override = rim_mat
		spin.add_child(spoke)
		wheels.append([pivot, spin])

# ---- networking -----------------------------------------------------------------------
func _process(delta: float) -> void:
	peer.poll()
	var st := peer.get_status()
	if st == StreamPeerTCP.STATUS_CONNECTED:
		if not connected:
			connected = true
			_set_status("Connected. Receiving the track ...")
		var avail := peer.get_available_bytes()
		if avail > 0:
			buf += peer.get_utf8_string(avail)
		var cut := buf.rfind("\n")
		if cut >= 0:
			var lines := buf.substr(0, cut).split("\n", false)
			buf = buf.substr(cut + 1)
			var latest = null
			for line in lines:
				var msg = JSON.parse_string(line)
				if msg == null:
					continue
				if msg["type"] == "track":
					track = msg
					_build_track()
					_build_car()
					_set_status("")
				elif msg["type"] == "state":
					latest = msg
			if latest != null:
				state = latest
				have_state = true
		input_timer += delta
		if input_timer >= 1.0 / 60.0 and not track.is_empty():
			input_timer = 0.0
			_send_input()
	elif st == StreamPeerTCP.STATUS_ERROR or st == StreamPeerTCP.STATUS_NONE:
		if connected:
			_set_status("Lost the physics server. Waiting for it to come back ...")
			connected = false
		retry_timer += delta
		if retry_timer > 1.0:  # retry once a second (the server may still be starting)
			retry_timer = 0.0
			peer = StreamPeerTCP.new()
			peer.connect_to_host(HOST, PORT)
	if have_state:
		_update_car(delta)
		_update_camera(delta)
		_update_hud()
		_update_weather()
		drive_time += delta
		var at_s := OS.get_environment("CARSIM_SHOT_AT_S")
		var shoot_now := drive_time > shot_after if at_s == "" else (float(state["s"]) > float(at_s) and float(state["s"]) < float(at_s) + 40.0 and drive_time > 2.0)
		if shot_path != "" and shoot_now:
			get_viewport().get_texture().get_image().save_png(shot_path)
			_send_line({"type": "quit"})
			get_tree().quit()

func _send_line(d: Dictionary) -> void:
	peer.put_data((JSON.stringify(d) + "\n").to_utf8_buffer())

func _pressed_once(key: String, down: bool) -> bool:
	var was: bool = prev_keys.get(key, false)
	prev_keys[key] = down
	return down and not was

func _send_input() -> void:
	var steer := 0.0
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		steer += 1.0
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		steer -= 1.0
	var throttle := 1.0 if (Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP)) else 0.0
	var brake := 1.0 if (Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN)) else 0.0
	var up := Input.is_key_pressed(KEY_E) or Input.is_key_pressed(KEY_SHIFT)
	var down := Input.is_key_pressed(KEY_Q) or Input.is_key_pressed(KEY_CTRL)
	var drs := Input.is_key_pressed(KEY_R)
	var pads := Input.get_connected_joypads()
	if pads.size() > 0:
		var j: int = pads[0]
		var ax := Input.get_joy_axis(j, JOY_AXIS_LEFT_X)
		if abs(ax) > 0.05:
			steer = -ax
		throttle = max(throttle, Input.get_joy_axis(j, JOY_AXIS_TRIGGER_RIGHT))
		brake = max(brake, Input.get_joy_axis(j, JOY_AXIS_TRIGGER_LEFT))
		up = up or Input.is_joy_button_pressed(j, JOY_BUTTON_RIGHT_SHOULDER) or Input.is_joy_button_pressed(j, JOY_BUTTON_A)
		down = down or Input.is_joy_button_pressed(j, JOY_BUTTON_LEFT_SHOULDER) or Input.is_joy_button_pressed(j, JOY_BUTTON_B)
		drs = drs or Input.is_joy_button_pressed(j, JOY_BUTTON_Y)
	_send_line({"type": "input", "steer": steer, "throttle": throttle, "brake": brake,
		"shift_up": up, "shift_down": down, "drs": drs})
	var pad0: int = pads[0] if pads.size() > 0 else -1
	if _pressed_once("reset", Input.is_key_pressed(KEY_BACKSPACE) or (pad0 >= 0 and Input.is_joy_button_pressed(pad0, JOY_BUTTON_BACK))):
		_send_line({"type": "reset"})
	if _pressed_once("restart", Input.is_key_pressed(KEY_F5)):
		_send_line({"type": "restart"})
	if _pressed_once("cam", Input.is_key_pressed(KEY_C) or (pad0 >= 0 and Input.is_joy_button_pressed(pad0, JOY_BUTTON_X))):
		cam_mode = (cam_mode + 1) % 3
	if _pressed_once("line", Input.is_key_pressed(KEY_L)):
		line_mesh.visible = not line_mesh.visible
	if _pressed_once("quit", Input.is_key_pressed(KEY_ESCAPE)):
		_send_line({"type": "quit"})
		get_tree().quit()

# ---- drawing the simulation state -------------------------------------------------------
func _car_basis() -> Basis:
	var yaw: float = state["yaw"]
	# The whole car (wheels included) follows the track's slope only; roll and suspension pitch
	# are applied to the body alone (see _update_car), so the wheels stay on the ground.
	return Basis(Vector3.UP, yaw - PI / 2) * Basis(Vector3.RIGHT, state["pitch"])

func _update_car(delta: float) -> void:
	var target := Transform3D(_car_basis(), sim_to_godot(state["x"], state["y"], state["z"]))
	# Light smoothing between 60 Hz updates and the render frame rate.
	var k: float = clamp(delta * 30.0, 0.0, 1.0)
	car_root.transform = car_root.transform.interpolate_with(target, k)
	body.rotation = Vector3(float(state.get("body_pitch", 0.0)), 0.0, float(state["roll"]))
	body.position.y = float(state.get("body_dz", 0.0))
	var steer: float = state["steer"]
	var spins: Array = state["wheels"]
	for i in 4:
		var pivot: Node3D = wheels[i][0]
		var spin: Node3D = wheels[i][1]
		pivot.rotation.y = steer if i < 2 else 0.0
		spin.rotation.x = -float(spins[i])

func _update_camera(delta: float) -> void:
	var t := car_root.global_transform
	var fwd := -t.basis.z
	fwd.y = 0
	fwd = fwd.normalized()
	if cam_mode == 0:  # chase: fixed distance behind, only the direction is smoothed
		cam_dir = cam_dir.lerp(fwd, clamp(delta * 5.0, 0.0, 1.0)).normalized()
		camera.global_position = t.origin - cam_dir * 8.5 + Vector3(0, 2.8, 0)
		camera.look_at(t.origin + cam_dir * 6.0 + Vector3(0, 1.0, 0))
		camera.fov = 70
	elif cam_mode == 1:  # cockpit
		camera.global_transform = Transform3D(t.basis, t.origin + t.basis * Vector3(0, 1.0, -0.3))
		camera.fov = 80
	else:  # TV: nearest camera spot, zoomed on the car
		var best = tv_points[0]
		for p in tv_points:
			if p.distance_to(t.origin) < best.distance_to(t.origin):
				best = p
		camera.global_position = best
		camera.look_at(t.origin + Vector3(0, 0.5, 0))
		camera.fov = clamp(2.0 * rad_to_deg(atan(6.0 / max(best.distance_to(t.origin), 1.0))), 6.0, 60.0)

func _update_weather() -> void:
	var r: float = state["rain"]
	rain.emitting = r > 0.2
	# Heavier rain = more visible streaks (CPU particles have a fixed count).
	var dm: StandardMaterial3D = (rain.mesh as QuadMesh).material
	dm.albedo_color.a = clamp(0.15 + r / 10.0, 0.15, 0.6)
	rain.global_position = camera.global_position + Vector3(0, 12, 0)
	# Looks damp whenever it rains, glossy as standing water builds.
	var wet: float = clamp(max(float(state["water"]) / 0.3, r / 3.0), 0.0, 1.0)
	asphalt.albedo_color = Color(0.16, 0.16, 0.17).lerp(Color(0.08, 0.08, 0.09), wet)
	asphalt.roughness = lerp(0.95, 0.2, wet)
	asphalt.metallic_specular = lerp(0.2, 0.6, wet)
	var cloud: float = clamp(r / 4.0, 0.0, 1.0)
	sky_mat.sky_top_color = Color(0.32, 0.52, 0.85).lerp(Color(0.45, 0.47, 0.5), cloud)
	sun.light_energy = lerp(1.2, 0.35, cloud)
	env.fog_density = lerp(0.0008, 0.006, cloud)

# ---- HUD ------------------------------------------------------------------------------
func _build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var make := func(pos: Vector2, size: int, color := Color.WHITE) -> Label:
		var l := Label.new()
		l.position = pos
		l.add_theme_font_size_override("font_size", size)
		l.add_theme_color_override("font_color", color)
		l.add_theme_color_override("font_outline_color", Color(0, 0, 0, 0.8))
		l.add_theme_constant_override("outline_size", 6)
		layer.add_child(l)
		return l
	hud["speed"] = make.call(Vector2(24, 14), 54)
	hud["gear"] = make.call(Vector2(24, 82), 22)
	hud["lap"] = make.call(Vector2(24, 116), 18)
	hud["tyres"] = make.call(Vector2(24, 146), 16)
	hud["info"] = make.call(Vector2(24, 172), 16)
	hud["gap"] = make.call(Vector2(1060, 14), 40)
	hud["help"] = make.call(Vector2(24, 680), 14, Color(1, 1, 1, 0.75))
	hud["status"] = make.call(Vector2(340, 300), 22)
	var bar := ProgressBar.new()
	bar.position = Vector2(150, 92)
	bar.size = Vector2(220, 14)
	bar.max_value = 15000
	bar.show_percentage = false
	layer.add_child(bar)
	hud["rpm"] = bar

func _set_status(text: String) -> void:
	hud["status"].text = text

func _update_hud() -> void:
	hud["speed"].text = "%d km/h" % int(state["v"])
	hud["gear"].text = "gear %d" % int(state["gear"])
	hud["rpm"].value = state["rpm"]
	var last = state["last"]
	var best = state["best"]
	hud["lap"].text = "lap %d   %.2f s   last %s   best %s" % [state["lap"], state["lap_time"],
		"-" if last == null else "%.2f" % last, "-" if best == null else "%.2f" % best]
	var ty: Array = state["tyres"]
	hud["tyres"].text = "tyres  %d  %d  /  %d  %d  C   %s" % [ty[0], ty[1], ty[2], ty[3], track["compound"]]
	var weather := "rain %.1f mm/h, water %.2f mm" % [state["rain"], state["water"]] if float(state["rain"]) > 0.05 or float(state["water"]) > 0.01 else "dry"
	hud["info"].text = "fuel %.1f kg   track %.0f C   %s%s" % [state["fuel"], state["track_temp"], weather,
		"   ON THE GRASS" if int(state["grass"]) >= 2 else ""]
	var gap = state["gap"]
	if gap == null:
		hud["gap"].text = ""
	else:
		hud["gap"].text = "%+.2f" % gap
		hud["gap"].add_theme_color_override("font_color", Color(0.4, 0.9, 0.5) if gap < 0 else Color(0.95, 0.45, 0.35))
	var who := ("Self-taught AI driving (reloads as it learns)" if track.get("learned", false) else "AI driving (watch mode)") if track["ai"] else "W/S or arrows: throttle/brake  A/D: steer  E/Q: shift  R: DRS"
	hud["help"].text = "%s   Backspace: back on track  C: camera (%s)  L: racing line  F5: restart  Esc: quit" % [who, CAM_NAMES[cam_mode]]
