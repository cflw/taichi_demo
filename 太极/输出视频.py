import taichi as ti
ti.init(arch = ti.gpu)
v画布 = ti.Vector.field(3, float, shape = (400, 400))
v视频管理 = ti.VideoManager(output_dir = "video", framerate = 24)	#输出路径不能有中文
@ti.func
def hsv2rgb(h: float, s: float, v: float) -> ti.Vector:
	hh = (h / 3.1415926 * 3.0) %6	#[0, 6)
	i = ti.floor(hh)
	ff = hh - i
	p = v * (1.0 - s)
	q = v * (1.0 - (s * ff))
	t = v * (1.0 - (s * (1.0 - ff)))

	r, g, b = 0.0, 0.0, 0.0
	if i == 0: r, g, b = v, t, p
	elif i == 1: r, g, b = q, v, p
	elif i == 2: r, g, b = p, v, t
	elif i == 3: r, g, b = p, q, v
	elif i == 4: r, g, b = t, p, v
	elif i == 5: r, g, b = v, p, q
	return ti.Vector([r, g, b])
@ti.kernel
def compute(t: float):
	for x, y in v画布:
		v画布[x, y] = hsv2rgb(t, 1, 1)
for frame in range(100):
	compute(frame / 24)
	v视频管理.write_frame(v画布)	#保存到 路径\frame\xxxxxx.png
v视频管理.make_video(gif = True, mp4 = True)	#默认都为True, 保存到 路径\路径.gif 和 路径\路径.mp4
print(f"保存到 {v视频管理.get_output_filename('参数')}")	#"保存到 路径\路径参数"