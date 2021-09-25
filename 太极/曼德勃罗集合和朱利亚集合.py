import taichi as ti
ti.init(arch=ti.gpu)
c窗口边长 = 800
c窗口尺寸 = (c窗口边长, c窗口边长)
c最大迭代数 = 20
c饱和度 = 0.8
c缩放 = 4	#缩小4倍
v画布1 = ti.Vector.field(3, dtype = float, shape = c窗口尺寸)	#曼德勃罗集合
v画布2 = ti.Vector.field(3, dtype = float, shape = c窗口尺寸)	#朱利亚集合
#常用函数
@ti.func
def complex_sqr(z)->ti.Vector:
	return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])
@ti.func
def cartesian(x: float, y: float)->ti.Vector:	#场索引到直角坐标
	return ti.Vector([x / c窗口边长 - 0.5, y / c窗口边长 - 0.5]) * c缩放
@ti.func
def color_hsv(h: float, s: float, v: float)->ti.Vector:
	#根据色环取颜色, 色相单位:弧度
	h = h / 3.1415926 * 3.0 % 6	#[0, 6)
	r = 0.0
	g = 0.0
	b = 0.0
	if h < 1:	#红->黄
		r = 1
		g = lerp(0, 1, h)
	elif h < 2:	#黄->绿
		r = lerp(1, 0, h - 1)
		g = 1
	elif h < 3:	#绿->青
		g = 1
		b = lerp(0, 1, h - 2)
	elif h < 4:	#青->蓝
		g = lerp(1, 0, h - 3)
		b = 1
	elif h < 5:	#蓝->紫
		b = 1
		r = lerp(0, 1, h - 4)
	else:	#紫->红
		b = lerp(1, 0, h - 5)
		r = 1
	r = lerp(1, r, s) * v
	g = lerp(1, g, s) * v
	b = lerp(1, b, s) * v
	return ti.Vector([r, g, b])
@ti.func
def lerp(a: float, b: float, t: float):
	return a + (b - a) * t
#核心
@ti.kernel
def mandelbrot(t: float):
	for i, j in v画布1:
		c = cartesian(i, j)
		v迭代数 = iter(c, c)
		v画布1[i, j] = color_hsv(ti.atan2(c[0], c[1]) + t, c饱和度, (c最大迭代数 - v迭代数) / c最大迭代数)
@ti.func
def iter(z, c)->int:
	#给定z, c, 返回迭代数
	v迭代数 = 0
	while z.norm() < 2 and v迭代数 < c最大迭代数:
		z = complex_sqr(z) + c
		v迭代数 += 1
	return v迭代数
@ti.kernel
def julia(cx: float, cy: float, t: float):
	c = ti.Vector([cx - 0.5, cy - 0.5]) * c缩放
	for i, j in v画布2:
		z = cartesian(i, j)
		v迭代数 = iter(z, c)
		v画布2[i, j] = color_hsv(ti.atan2(z[0], z[1]) + t, c饱和度, (c最大迭代数 - v迭代数) / c最大迭代数)
#窗口
t = 0
v窗口1 = ti.GUI("曼德勃罗集合", res = c窗口尺寸)
v窗口2 = ti.GUI("朱利亚集合", res = c窗口尺寸)
while v窗口1.running and v窗口2.running:
	t += 1 / 60.0
	x, y = v窗口1.get_cursor_pos()
	mandelbrot(t)
	julia(x, y, t)
	v窗口1.set_image(v画布1)
	v窗口2.set_image(v画布2)
	v窗口1.show()
	v窗口2.show()
else:
	v窗口1.running = False
	v窗口2.running = False