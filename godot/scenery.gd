extends RefCounted
## Scenery and a shaped car for the carsim 3D view, kept cheap for a weak laptop:
## a few hundred triangles for the car, the trees drawn in one batch (MultiMesh, no shadows),
## grandstands and the start gantry from simple boxes. CARSIM_GFX=low skips the scenery.
##
## Simulation axes: x east, y north, z up. Godot axes: x right, y up, -z forward.

const TREE_EVERY := 9  # samples (~m) between tree spots along each side of the lap
const TREE_KEEP := 0.55  # share of the spots that get a tree
const TREE_CLEAR := 14.0  # m: no tree closer than this to any part of the track (other bends included)
const STAND_LEN := 26  # samples per grandstand block

static func mat(color: Color, rough := 0.8, metal := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = rough
	m.metallic = metal
	return m

static func vmat(rough := 0.9) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.vertex_color_use_as_albedo = true
	m.roughness = rough
	m.metallic_specular = 0.1
	return m

static func edge(t: Dictionary, i: int, offset: float, lift: float) -> Vector3:
	var h: float = t["heading"][i]
	var x: float = float(t["x"][i]) - offset * sin(h)
	var y: float = float(t["y"][i]) + offset * cos(h)
	return Vector3(x, float(t["z"][i]) + lift, -y)

static func runs(code: String) -> Array:
	var out := []
	var at := 0
	var k := 0
	while k < code.length():
		var c := code[k]
		k += 1
		var num := ""
		while k < code.length() and code[k] >= "0" and code[k] <= "9":
			num += code[k]
			k += 1
		out.append([c, at, at + int(num)])
		at += int(num)
	return out

## How far out the mapped ground ends on each side (the barrier, else the mapped metres): [left[], right[]].
static func outer(t: Dictionary) -> Array:
	var n: int = t["x"].size()
	var res := []
	for side in ["surf_left", "surf_right"]:
		var o := []
		o.resize(n)
		for i in n:
			o[i] = 40.0
			if t.has(side):
				var r := runs(t[side][i])
				o[i] = float(r[r.size() - 1][2])
				for q in r:
					if q[0] == "B":
						o[i] = float(q[1])
						break
		res.append(o)
	return res

## A coarse grid of the track's centreline, to keep trees off every part of the track.
static func grid(t: Dictionary, cell: float) -> Dictionary:
	var g := {}
	var n: int = t["x"].size()
	for i in range(0, n, 3):
		var key := Vector2i(int(floor(float(t["x"][i]) / cell)), int(floor(float(t["y"][i]) / cell)))
		if not g.has(key):
			g[key] = []
		g[key].append(Vector3(float(t["x"][i]), float(t["y"][i]), max(float(t["width_left"][i]), float(t["width_right"][i]))))
	return g

static func clear_of_track(g: Dictionary, cell: float, x: float, y: float, clear: float) -> bool:
	var cx := int(floor(x / cell))
	var cy := int(floor(y / cell))
	for dx in [-1, 0, 1]:
		for dy in [-1, 0, 1]:
			var key := Vector2i(cx + dx, cy + dy)
			if g.has(key):
				for p in g[key]:
					if Vector2(x - p.x, y - p.y).length() < p.z + clear:
						return false
	return true

# ---- trees ---------------------------------------------------------------------------
static func _tree_mesh() -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var sides := 7
	var trunk := Color(0.33, 0.24, 0.16)
	var leaf := [Color(0.16, 0.34, 0.14), Color(0.13, 0.29, 0.12)]
	# trunk: an open prism 0 .. 2.2 m
	for k in sides:
		var a0 := TAU * k / sides
		var a1 := TAU * (k + 1) / sides
		var p0 := Vector3(cos(a0) * 0.22, 0, sin(a0) * 0.22)
		var p1 := Vector3(cos(a1) * 0.22, 0, sin(a1) * 0.22)
		st.set_color(trunk)
		for v in [p0, p1, p0 + Vector3(0, 2.2, 0), p1, p1 + Vector3(0, 2.2, 0), p0 + Vector3(0, 2.2, 0)]:
			st.add_vertex(v)
	# crown: two stacked cones
	var cones := [[1.6, 2.4, 6.0], [4.0, 1.7, 8.2]]  # base height, radius, tip height
	for c in cones.size():
		var cn: Array = cones[c]
		st.set_color(leaf[c])
		for k in sides:
			var a0 := TAU * k / sides
			var a1 := TAU * (k + 1) / sides
			var b0 := Vector3(cos(a0) * cn[1], cn[0], sin(a0) * cn[1])
			var b1 := Vector3(cos(a1) * cn[1], cn[0], sin(a1) * cn[1])
			var tip := Vector3(0, cn[2], 0)
			for v in [b0, b1, tip, b1, b0, Vector3(0, cn[0], 0)]:
				st.add_vertex(v)
	st.generate_normals()
	return st.commit()

const TV_CLEAR := 30.0  # m: no trees round the TV camera spots (they would fill the picture)

static func build_trees(parent: Node3D, t: Dictionary, keep_clear: Array = []) -> int:
	var n: int = t["x"].size()
	var out := outer(t)
	var cell := 40.0
	var g := grid(t, cell)
	var rng := RandomNumberGenerator.new()
	rng.seed = 1953
	var xf := []
	for s in 2:
		var side := 1.0 if s == 0 else -1.0
		var w: Array = t["width_left"] if s == 0 else t["width_right"]
		for i in range(0, n, TREE_EVERY):
			if rng.randf() > TREE_KEEP:
				continue
			var off := float(w[i]) + float(out[s][i]) + rng.randf_range(6.0, 45.0)
			var h: float = t["heading"][i]
			var x: float = float(t["x"][i]) - side * off * sin(h)
			var y: float = float(t["y"][i]) + side * off * cos(h)
			if not clear_of_track(g, cell, x, y, TREE_CLEAR):
				continue
			var near_cam := false
			for c in keep_clear:
				if Vector2(c.x - x, c.z + y).length() < TV_CLEAR:  # Godot (x, z) = sim (x, -y)
					near_cam = true
					break
			if near_cam:
				continue
			var sc := rng.randf_range(0.75, 1.35)
			var b := Basis(Vector3.UP, rng.randf() * TAU).scaled(Vector3(sc, sc * rng.randf_range(0.85, 1.2), sc))
			xf.append(Transform3D(b, Vector3(x, float(t["z"][i]) - 0.05, -y)))
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = _tree_mesh()
	mm.instance_count = xf.size()
	for k in xf.size():
		mm.set_instance_transform(k, xf[k])
	var mmi := MultiMeshInstance3D.new()
	mmi.multimesh = mm
	mmi.material_override = vmat(1.0)
	mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	parent.add_child(mmi)
	return xf.size()

# ---- grandstands and the start gantry ---------------------------------------------------
static func _box(parent: Node3D, size: Vector3, pos: Vector3, basis: Basis, m: Material, shadow := true) -> void:
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	mi.material_override = m
	mi.transform = Transform3D(basis, pos)
	if not shadow:
		mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	parent.add_child(mi)

## Grandstands along the start/finish straight, behind the barrier on both sides where there is room.
static func build_stands(parent: Node3D, t: Dictionary) -> void:
	var n: int = t["x"].size()
	var out := outer(t)
	var concrete := mat(Color(0.62, 0.62, 0.6), 0.9)
	var roof := mat(Color(0.85, 0.86, 0.88), 0.5, 0.2)
	var seat_colors := [Color(0.12, 0.25, 0.6), Color(0.75, 0.1, 0.1), Color(0.9, 0.9, 0.9), Color(0.1, 0.45, 0.25)]
	var blocks := [n - 3 * STAND_LEN - 20, n - 2 * STAND_LEN - 15, n - STAND_LEN - 10, 10, STAND_LEN + 15]
	for s in 2:
		var side := 1.0 if s == 0 else -1.0
		var w: Array = t["width_left"] if s == 0 else t["width_right"]
		for bi in blocks.size():
			var i0: int = (int(blocks[bi]) + n) % n
			var i1: int = (i0 + STAND_LEN) % n
			var im: int = (i0 + STAND_LEN / 2) % n
			var h0: float = t["heading"][i0]
			var h1: float = t["heading"][i1]
			if abs(wrapf(h1 - h0, -PI, PI)) > 0.08:  # only along a straight
				continue
			var gap := float(out[s][im]) + 4.0
			if gap > 45.0:
				continue
			var a := edge(t, i0, side * (float(w[i0]) + gap), 0.0)
			var b := edge(t, i1, side * (float(w[i1]) + gap), 0.0)
			var along := b - a
			var length := Vector2(along.x, along.z).length()
			var fwd := Vector3(along.x, 0, along.z).normalized()
			var outw := Vector3(-fwd.z, 0, fwd.x) * (-side)  # away from the track
			if (edge(t, im, side * 100.0, 0.0) - edge(t, im, 0.0, 0.0)).dot(outw) < 0:
				outw = -outw
			var basis := Basis(outw, Vector3.UP, -fwd).orthonormalized()
			var mid := (a + b) / 2.0
			var seat := mat(seat_colors[(bi + s) % seat_colors.size()], 0.7)
			for row in 7:
				var p := mid + outw * (1.0 + row * 1.1) + Vector3(0, 0.5 + row * 0.75, 0)
				_box(parent, Vector3(1.1, 0.75 + row * 0.0, length), p - Vector3(0, 0.375, 0), basis, concrete, false)
				_box(parent, Vector3(0.7, 0.35, length - 1.0), p + Vector3(0, 0.18, 0) - outw * 0.15, basis, seat, false)
			_box(parent, Vector3(0.4, 7.0, length), mid + outw * 8.6 + Vector3(0, 3.0, 0), basis, concrete)
			_box(parent, Vector3(9.5, 0.25, length + 1.0), mid + outw * 4.4 + Vector3(0, 7.2, 0), basis * Basis(Vector3.FORWARD, 0.06), roof)

static func build_gantry(parent: Node3D, t: Dictionary) -> void:
	var wl := float(t["width_left"][0]) + 2.5
	var wr := float(t["width_right"][0]) + 2.5
	var a := edge(t, 0, wl, 0.0)
	var b := edge(t, 0, -wr, 0.0)
	var grey := mat(Color(0.2, 0.2, 0.22), 0.6, 0.4)
	var across := b - a
	var basis := Basis(Vector3.UP, atan2(-across.z, across.x))
	_box(parent, Vector3(0.5, 7.0, 0.5), a + Vector3(0, 3.5, 0), basis, grey)
	_box(parent, Vector3(0.5, 7.0, 0.5), b + Vector3(0, 3.5, 0), basis, grey)
	_box(parent, Vector3(across.length() + 0.5, 1.0, 0.6), (a + b) / 2.0 + Vector3(0, 7.2, 0), basis, grey)
	var red := mat(Color(0.9, 0.05, 0.05), 0.3)
	red.emission_enabled = true
	red.emission = Color(0.5, 0.0, 0.0)
	for k in 5:
		var p := (a + b) / 2.0 + Vector3(0, 7.2, 0) + across.normalized() * (k - 2) * 0.9
		_box(parent, Vector3(0.45, 0.45, 0.2), p + Vector3(0, 0, 0.35).rotated(Vector3.UP, basis.get_euler().y), basis, red, false)

## A darker band where the cars run (rubber), every 2nd sample of the racing line.
static func build_rubber(parent: Node3D, t: Dictionary) -> void:
	if not t.has("line_x"):
		return
	var lx: Array = t["line_x"]
	var ly: Array = t["line_y"]
	var zs: Array = t["z"]
	var n := lx.size()
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var half := 0.95
	for i in range(0, n, 2):
		var j := (i + 2) % n
		var k := (j + 2) % n
		var d0 := Vector2(float(lx[j]) - float(lx[i]), float(ly[j]) - float(ly[i])).normalized()
		var d1 := Vector2(float(lx[k]) - float(lx[j]), float(ly[k]) - float(ly[j])).normalized()
		var n0 := Vector2(-d0.y, d0.x) * half
		var n1 := Vector2(-d1.y, d1.x) * half
		var z0 := float(zs[i]) + 0.006
		var z1 := float(zs[j]) + 0.006
		var a0 := Vector3(float(lx[i]) + n0.x, z0, -(float(ly[i]) + n0.y))
		var a1 := Vector3(float(lx[i]) - n0.x, z0, -(float(ly[i]) - n0.y))
		var b0 := Vector3(float(lx[j]) + n1.x, z1, -(float(ly[j]) + n1.y))
		var b1 := Vector3(float(lx[j]) - n1.x, z1, -(float(ly[j]) - n1.y))
		for v in [a0, a1, b0, a1, b1, b0]:
			st.add_vertex(v)
	st.generate_normals()
	var mi := MeshInstance3D.new()
	mi.mesh = st.commit()
	var m := mat(Color(0.115, 0.115, 0.12), 0.85)
	m.metallic_specular = 0.25
	mi.material_override = m
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	parent.add_child(mi)

# ---- the car ----------------------------------------------------------------------------
## A lofted body: cross-sections [z, half width, bottom, top, x centre] joined by quads (forward is -z).
static func _loft(sections: Array) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var ring := func(s: Array) -> Array:
		var z: float = s[0]
		var w: float = s[1]
		var lo: float = s[2]
		var hi: float = s[3]
		var cx: float = s[4] if s.size() > 4 else 0.0
		var mid := lo + (hi - lo) * 0.45
		return [Vector3(cx - w * 0.8, lo, z), Vector3(cx + w * 0.8, lo, z), Vector3(cx + w, mid, z),
				Vector3(cx + w * 0.62, hi, z), Vector3(cx - w * 0.62, hi, z), Vector3(cx - w, mid, z)]
	for k in sections.size() - 1:
		var r0: Array = ring.call(sections[k])
		var r1: Array = ring.call(sections[k + 1])
		for e in 6:
			var f := (e + 1) % 6
			for v in [r0[e], r1[e], r0[f], r0[f], r1[e], r1[f]]:
				st.add_vertex(v)
	for cap in [[sections[0], false], [sections[sections.size() - 1], true]]:
		var r: Array = ring.call(cap[0])
		for e in range(1, 5):
			var tri := [r[0], r[e], r[e + 1]] if cap[1] else [r[0], r[e + 1], r[e]]
			for v in tri:
				st.add_vertex(v)
	st.generate_normals()
	return st.commit()

static func _mesh(parent: Node3D, mesh: Mesh, m: Material, pos := Vector3.ZERO, rot := Vector3.ZERO) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = m
	mi.position = pos
	mi.rotation = rot
	parent.add_child(mi)
	return mi

static func _rod(parent: Node3D, p0: Vector3, p1: Vector3, r: float, m: Material) -> void:
	var cm := CylinderMesh.new()
	cm.top_radius = r
	cm.bottom_radius = r
	cm.height = (p1 - p0).length()
	cm.radial_segments = 8
	var mi := MeshInstance3D.new()
	mi.mesh = cm
	mi.material_override = m
	var up := (p1 - p0).normalized()
	var side := up.cross(Vector3.FORWARD)
	if side.length() < 0.01:
		side = up.cross(Vector3.RIGHT)
	side = side.normalized()
	mi.transform = Transform3D(Basis(side, up, side.cross(up)).orthonormalized(), (p0 + p1) / 2.0)
	parent.add_child(mi)

## The car body (on `body`, which pitches and rolls) and four wheels; returns [[steer pivot, spin node] x4].
static func build_car(car_root: Node3D, body: Node3D, t: Dictionary) -> Array:
	var a: float = t["a"]
	var b: float = t["b"]
	var tf: float = t["track_front"]
	var trr: float = t["track_rear"]
	var r: float = t["wheel_radius"]
	var paint := mat(Color(0.72, 0.05, 0.04), 0.3, 0.35)
	var accent := mat(Color(0.95, 0.95, 0.95), 0.35, 0.1)
	var carbon := mat(Color(0.07, 0.07, 0.08), 0.45, 0.2)
	# monocoque, nose to gearbox
	_mesh(body, _loft([
		[-a - 1.0, 0.07, 0.20, 0.30], [-a - 0.4, 0.13, 0.19, 0.38], [-a + 0.4, 0.2, 0.17, 0.48],
		[-0.55, 0.3, 0.12, 0.60], [-0.1, 0.34, 0.11, 0.58], [0.45, 0.34, 0.11, 0.66],
		[0.75, 0.3, 0.12, 0.98], [1.35, 0.25, 0.14, 0.82], [b - 0.2, 0.16, 0.2, 0.52], [b + 0.35, 0.1, 0.25, 0.42]]), paint)
	# white stripe on the nose
	_mesh(body, _loft([[-a - 0.95, 0.03, 0.29, 0.305], [-a + 0.35, 0.06, 0.465, 0.485], [-0.6, 0.08, 0.58, 0.605]]), accent)
	# sidepods
	for sx in [-1.0, 1.0]:
		_mesh(body, _loft([
			[-0.35, 0.14, 0.14, 0.40, sx * 0.55], [0.0, 0.24, 0.12, 0.52, sx * 0.6], [0.7, 0.22, 0.12, 0.48, sx * 0.55],
			[1.4, 0.13, 0.14, 0.36, sx * 0.42], [b - 0.1, 0.06, 0.18, 0.26, sx * 0.3]]), paint)
	# floor
	_mesh(body, _loft([[-0.9, 0.55, 0.05, 0.08], [0.2, 0.72, 0.05, 0.08], [b - 0.1, 0.55, 0.05, 0.08]]), carbon)
	# halo: a centre pillar and a hoop round the cockpit
	var halo := [Vector3(0, 0.88, -0.05), Vector3(0.24, 0.88, 0.05), Vector3(0.3, 0.86, 0.3), Vector3(0.24, 0.78, 0.5)]
	_rod(body, Vector3(0, 0.60, -0.42), Vector3(0, 0.88, -0.05), 0.025, carbon)
	for sx in [-1.0, 1.0]:
		for k in halo.size() - 1:
			_rod(body, halo[k] * Vector3(sx, 1, 1), halo[k + 1] * Vector3(sx, 1, 1), 0.025, carbon)
	# driver
	var hm := SphereMesh.new()
	hm.radius = 0.14
	hm.height = 0.28
	hm.radial_segments = 12
	hm.rings = 8
	_mesh(body, hm, mat(Color(0.95, 0.85, 0.1), 0.25), Vector3(0, 0.7, 0.12))
	# front wing: two elements, endplates
	var fw := BoxMesh.new()
	fw.size = Vector3(1.95, 0.03, 0.32)
	_mesh(body, fw, carbon, Vector3(0, 0.1, -a - 0.75))
	var fw2 := BoxMesh.new()
	fw2.size = Vector3(1.8, 0.025, 0.22)
	_mesh(body, fw2, paint, Vector3(0, 0.17, -a - 0.6), Vector3(0.35, 0, 0))
	for sx in [-1.0, 1.0]:
		var ep := BoxMesh.new()
		ep.size = Vector3(0.025, 0.2, 0.5)
		_mesh(body, ep, carbon, Vector3(sx * 0.97, 0.16, -a - 0.7))
	# rear wing: main plane, flap, endplates, pillar
	var rw := BoxMesh.new()
	rw.size = Vector3(1.0, 0.035, 0.3)
	_mesh(body, rw, carbon, Vector3(0, 0.86, b + 0.42), Vector3(0.12, 0, 0))
	var rf := BoxMesh.new()
	rf.size = Vector3(1.0, 0.03, 0.2)
	_mesh(body, rf, paint, Vector3(0, 0.98, b + 0.5), Vector3(0.5, 0, 0))
	for sx in [-1.0, 1.0]:
		var rep := BoxMesh.new()
		rep.size = Vector3(0.025, 0.55, 0.62)
		_mesh(body, rep, accent, Vector3(sx * 0.51, 0.8, b + 0.42))
	_rod(body, Vector3(0, 0.45, b + 0.3), Vector3(0, 0.86, b + 0.38), 0.03, carbon)
	# mirrors
	for sx in [-1.0, 1.0]:
		var mb := BoxMesh.new()
		mb.size = Vector3(0.13, 0.06, 0.04)
		_mesh(body, mb, paint, Vector3(sx * 0.42, 0.66, -0.45))
	# wheels: steer pivot (front) -> spinning tyre, rim and a marker so the spin shows
	var tyre_mat := mat(Color(0.05, 0.05, 0.05), 0.9)
	var rim_mat := mat(Color(0.18, 0.18, 0.2), 0.35, 0.7)
	var band := mat(Color(0.95, 0.8, 0.1), 0.6)  # medium-compound band on the sidewall
	var spots := [Vector3(-tf / 2, r, -a), Vector3(tf / 2, r, -a), Vector3(-trr / 2, r, b), Vector3(trr / 2, r, b)]
	var wheels := []
	for k in 4:
		var pivot := Node3D.new()
		pivot.position = spots[k]
		car_root.add_child(pivot)
		var spin := Node3D.new()
		pivot.add_child(spin)
		var wdt := 0.38 if k >= 2 else 0.32
		var cm := CylinderMesh.new()
		cm.top_radius = r
		cm.bottom_radius = r
		cm.height = wdt
		cm.radial_segments = 24
		_mesh(spin, cm, tyre_mat, Vector3.ZERO, Vector3(0, 0, PI / 2))
		var rm := CylinderMesh.new()
		rm.top_radius = r * 0.62
		rm.bottom_radius = r * 0.62
		rm.height = wdt + 0.01
		rm.radial_segments = 16
		_mesh(spin, rm, rim_mat, Vector3.ZERO, Vector3(0, 0, PI / 2))
		var bd := CylinderMesh.new()
		bd.top_radius = r * 0.8
		bd.bottom_radius = r * 0.8
		bd.height = wdt + 0.004
		bd.radial_segments = 24
		_mesh(spin, bd, band, Vector3.ZERO, Vector3(0, 0, PI / 2))
		var mk := BoxMesh.new()
		mk.size = Vector3(wdt + 0.02, 0.05, r * 1.15)
		_mesh(spin, mk, mat(Color(0.6, 0.6, 0.62), 0.4, 0.6), Vector3.ZERO)
		wheels.append([pivot, spin])
	return wheels
