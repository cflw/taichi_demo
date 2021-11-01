import taichi as ti
import time
ti.init(arch = ti.gpu)
PI = 3.14159265
c窗口边长 = 800
c窗口半边长 = 800 // 2
c窗口尺寸 = (c窗口边长, c窗口边长)
c烟花粒子数量 = 500	#放一次烟花有多少个粒子
c粒子上限 = 10000
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
t粒子 = ti.types.struct(m位置 = t向量2, m速度 = t向量2, m颜色 = t向量3, m半径 = float, m寿命 = float, m时间 = float)
v图像 = ti.Vector.field(3, dtype = float, shape = c窗口尺寸)
v粒子 = t粒子.field(shape = c粒子上限)
v新粒子索引 = ti.field(float, shape = ())
@ti.func
def new_particle(x: float, y: float):	#新粒子
	v方向 = ti.random() * PI * 2
	v位置 = ti.Vector([x, y])
	v速度 = ti.Vector([ti.sin(v方向), ti.cos(v方向)]) * (ti.random() * 500 + 100)
	v颜色 = ti.Vector([ti.random(), ti.random(), ti.random()])	#[0, 1)
	v半径 = ti.random() * 10 + 10
	v寿命 = ti.random() * 2 + 0.1
	return t粒子(m位置 = v位置, m速度 = v速度, m颜色 = v颜色, m半径 = v半径, m寿命 = v寿命, m时间 = 0)
@ti.kernel
def new_particles(x: float, y: float):	#新粒子s
	if v新粒子索引[None] + c烟花粒子数量 >= c粒子上限:
		v新粒子索引[None] = 0
	v新粒子索引结束 = v新粒子索引[None] + c烟花粒子数量
	for i in range(v新粒子索引[None], v新粒子索引结束):
		v粒子[i] = new_particle(x, y)
	v新粒子索引[None] = v新粒子索引结束
@ti.kernel
def init():
	v新粒子索引[None] = 0
	for i in v粒子:
		v粒子[i].m寿命 = 0
		v粒子[i].m时间 = 0
@ti.kernel
def compute(dt: float):
	for i in v粒子:
		if v粒子[i].m寿命 > v粒子[i].m时间:
			v粒子[i].m位置 += v粒子[i].m速度 * dt
			v粒子[i].m时间 += dt
			v粒子边界 = c窗口半边长 + v粒子[i].m半径
			if ti.abs(v粒子[i].m位置[0]) > v粒子边界 or ti.abs(v粒子[i].m位置[1]) > v粒子边界:	#超出边界
				v粒子[i].m寿命 = 0	#置为无效
@ti.kernel
def paint():
	for i in v粒子:
		if v粒子[i].m寿命 > v粒子[i].m时间:
			a0 = 1.0 - v粒子[i].m时间 / v粒子[i].m寿命	#根据寿命计算不透明度
			x0 = v粒子[i].m位置[0] + c窗口半边长	#到图像坐标
			y0 = v粒子[i].m位置[1] + c窗口半边长
			v粒子半径 = v粒子[i].m半径
			l = max(int(x0 - v粒子半径), 0)	#左
			r = min(int(x0 + v粒子半径) + 1, c窗口边长)	#右
			t = min(int(y0 + v粒子半径) + 1, c窗口边长)	#上
			b = max(int(y0 - v粒子半径), 0)	#下
			for x1, y1 in ti.ndrange((l, r), (b, t)):	#遍历粒子周围图像像素
				x2 = float(x1) - x0	#相对粒子的坐标
				y2 = float(y1) - y0
				v距离 = ti.sqrt(x2 * x2 + y2 * y2)	#到粒子中心的距离
				a1 = 1.0 - v距离 / v粒子半径	#根据距离计算不透明度
				if a1 > 0:
					v图像[x1, y1] += v粒子[i].m颜色 * a0 * a1
v窗口 = ti.ui.Window("烟花", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
v上次时间 = time.time()
v上次左键 = False
v这次左键 = False
while v窗口.running:
	#按键
	v上次左键 = v这次左键
	v这次左键 = v窗口.is_pressed(ti.ui.LMB)
	if v上次左键 == False and v这次左键 == True:	#刚按下
		x, y = v窗口.get_cursor_pos()
		new_particles(x * c窗口边长 - c窗口半边长, y * c窗口边长 - c窗口半边长)	#转坐标
	#时间
	if (v这次时间 := time.time()) > v上次时间:
		v时间间隔 = v这次时间 - v上次时间
		compute(v时间间隔)
		v上次时间 = v这次时间
	#绘制
	v图像.fill(0)
	paint()
	v画布.set_image(v图像)
	v窗口.show()