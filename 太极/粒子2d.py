import taichi as ti
import time
ti.init(arch = ti.gpu)
PI = 3.14159265
c窗口边长 = 800
c窗口半边长 = 800 // 2
c窗口尺寸 = (c窗口边长, c窗口边长)
c粒子数量 = 2000
c粒子半径 = 20
c粒子边界 = c窗口半边长 + c粒子半径
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
t粒子 = ti.types.struct(m位置 = t向量2, m速度 = t向量2, m颜色 = t向量3)
v图像 = ti.Vector.field(3, dtype = float, shape = c窗口尺寸)
v粒子 = t粒子.field(shape = c粒子数量)
@ti.func
def new_particle():	#新粒子
	v方向 = ti.random() * PI * 2
	v位置 = ti.Vector([0, 0])
	v速度 = ti.Vector([ti.sin(v方向), ti.cos(v方向)]) * (ti.random() * 100 + 1)
	v颜色 = ti.Vector([ti.random(), ti.random(), ti.random()])	#[0, 1)
	return t粒子(m位置 = v位置, m速度 = v速度, m颜色 = v颜色)
@ti.kernel
def init():
	for i in v粒子:
		v粒子[i] = new_particle()
@ti.kernel
def compute(dt: float):
	for i in v粒子:
		v粒子[i].m位置 += v粒子[i].m速度 * dt
		if ti.abs(v粒子[i].m位置[0]) > c粒子边界 or ti.abs(v粒子[i].m位置[1]) > c粒子边界:	#超出边界
			v粒子[i] = new_particle()
@ti.kernel
def paint():
	for i in v粒子:
		x0 = v粒子[i].m位置[0] + c窗口半边长	#到图像坐标
		y0 = v粒子[i].m位置[1] + c窗口半边长
		l = max(int(x0) - c粒子半径, 0)	#左
		r = min(int(x0) + c粒子半径 + 1, c窗口边长)	#右
		t = min(int(y0) + c粒子半径 + 1, c窗口边长)	#上
		b = max(int(y0) - c粒子半径, 0)	#下
		for x1, y1 in ti.ndrange((l, r), (b, t)):	#遍历粒子周围图像像素
			x2 = float(x1) - x0	#相对粒子的坐标
			y2 = float(y1) - y0
			v距离 = ti.sqrt(x2 * x2 + y2 * y2)	#到粒子中心的距离
			a = 1.0 - v距离 / c粒子半径	#不透明度
			if a > 0:
				v图像[x1, y1] += v粒子[i].m颜色 * a
v窗口 = ti.ui.Window("粒子", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
v上次时间 = time.time()
while v窗口.running:
	if (v这次时间 := time.time()) > v上次时间:
		v时间间隔 = v这次时间 - v上次时间
		compute(v时间间隔)
		v上次时间 = v这次时间
	v图像.fill(0)
	paint()
	v画布.set_image(v图像)
	v窗口.show()