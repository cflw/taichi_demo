import taichi as ti
import time
ti.init(arch = ti.gpu)
PI = 3.14159265
c窗口边长 = 800
c窗口半径 = (c窗口边长, c窗口边长)
c数量 = 100
c半径 = 0.02
c直径 = c半径 * 2
c质量 = 0.1
t向量2 = ti.types.vector(2, float)
t物体 = ti.types.struct(m位置 = t向量2, m速度 = t向量2, m质量 = float)
v物体 = t物体.field(shape = c数量)
v变化 = t物体.field(shape = c数量)
v圆心 = ti.Vector.field(2, dtype = float, shape = c数量)
@ti.kernel
def init():
	for i in v物体:
		v方向 = ti.random() * PI * 2
		v物体[i].m位置 = ti.Vector([ti.random(), ti.random()])
		v物体[i].m速度 = ti.Vector([ti.sin(v方向), ti.cos(v方向)]) * 0.2
		v物体[i].m质量 = c质量
@ti.kernel
def compute(dt: float):
	#位置
	for i in v物体:
		v物体[i].m位置 += v物体[i].m速度 * dt
		if v物体[i].m位置.x < 0:	#边界反弹
			v物体[i].m位置.x = 0
			v物体[i].m速度.x = -v物体[i].m速度.x
		elif v物体[i].m位置.x > 1:
			v物体[i].m位置.x = 1
			v物体[i].m速度.x = -v物体[i].m速度.x
		if v物体[i].m位置.y < 0:
			v物体[i].m位置.y = 0
			v物体[i].m速度.y = -v物体[i].m速度.y
		elif v物体[i].m位置.y > 1:
			v物体[i].m位置.y = 1
			v物体[i].m速度.y = -v物体[i].m速度.y
		#清除变化
		v变化[i].m位置 = ti.Vector([0, 0])
		v变化[i].m速度 = v物体[i].m速度
	#碰撞
	for i in v物体:
		v速度i = v物体[i].m速度
		v质量i = v物体[i].m质量
		for j in range(i):
			v速度j = v物体[j].m速度
			v质量j = v物体[j].m质量
			v位置差 = v物体[j].m位置 - v物体[i].m位置
			v距离 = v位置差.norm()
			if v距离 <= c直径:	#发生碰撞
				#稍微拉开一点距离,防止下一帧再次发生碰撞
				v拉开距离 = v位置差 * (1 - v距离 / c直径)
				v变化[i].m位置 -= v拉开距离
				v变化[j].m位置 += v拉开距离
				#碰撞后速度,由动量守恒定律和能量守恒定律得到
				v质量比i = v质量i / v质量j	#不会算二维空间的质量差,先用比例代替
				v质量比j = v质量j / v质量i
				v方向 = v位置差 / v距离
				v法线 = ti.Vector([v方向.y, -v方向.x])
				v变化[i].m速度 = v速度i.dot(v法线)*v法线 * v质量比i + v速度j.dot(v方向)*v方向 * v质量比j
				v变化[j].m速度 = v速度i.dot(v方向)*v方向 * v质量比i + v速度j.dot(v法线)*v法线 * v质量比j
				# print(v变化[i].m速度.norm(), v变化[j].m速度.norm())
	#应用变化
	for i in v物体:
		v物体[i].m位置 += v变化[i].m位置
		v物体[i].m速度 = v变化[i].m速度
		v圆心[i] = v物体[i].m位置
init()
v窗口 = ti.ui.Window("碰撞2d", res = c窗口半径)
v画布 = v窗口.get_canvas()
v上次时间 = time.time()
while v窗口.running:
	#时间
	if (v这次时间 := time.time()) > v上次时间:
		v时间间隔 = v这次时间 - v上次时间
		compute(v时间间隔)
		v上次时间 = v这次时间
	v画布.set_background_color((0, 0, 0))
	v画布.circles(v圆心, c半径, (1, 1, 1))
	v窗口.show()