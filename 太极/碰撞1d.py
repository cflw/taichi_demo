import taichi as ti
import time
ti.init(arch = ti.gpu)
#窗口
c窗口边长 = 800
c窗口半径 = (c窗口边长, c窗口边长)
c最大时间差 = 0.01
#物理参数
c数量 = 5
c半径 = 0.01
c直径 = c半径 * 2
c质量 = 0.1
t物体 = ti.types.struct(
	m位置 = float, 
	m速度 = float, 
	m质量 = float
)
v物体 = t物体.field(shape = c数量)
v变化 = t物体.field(shape = c数量)
v圆心 = ti.Vector.field(2, dtype = float, shape = c数量)
#计算
@ti.kernel
def init():
	for i in v物体:
		v物体[i].m位置 = ti.random()
		v物体[i].m速度 = (ti.random() - 0.5) * 0.5
		v物体[i].m质量 = c质量
		v圆心[i][1] = 0.5
@ti.kernel
def compute(dt: float):
	#位置
	for i in v物体:
		v物体[i].m位置 += v物体[i].m速度 * dt
		if v物体[i].m位置 < 0:	#边界反弹
			v物体[i].m位置 = 0
			v物体[i].m速度 = -v物体[i].m速度
		elif v物体[i].m位置 > 1:
			v物体[i].m位置 = 1
			v物体[i].m速度 = -v物体[i].m速度
		#清除变化
		v变化[i].m位置 = 0
		v变化[i].m速度 = 0
	#碰撞
	for i in v物体:
		v速度i = v物体[i].m速度
		v质量i = v物体[i].m质量
		for j in range(i):
			v速度j = v物体[j].m速度
			v质量j = v物体[j].m质量
			v位置差 = v物体[j].m位置 - v物体[i].m位置
			v距离 = ti.abs(v位置差)
			if v距离 <= c直径:	#发生碰撞
				#稍微拉开一点距离,防止下一帧再次发生碰撞
				v拉开距离 = v位置差 * (1 - v距离 / c直径)
				v变化[i].m位置 -= v拉开距离
				v变化[j].m位置 += v拉开距离
				#碰撞后速度,由动量守恒定律和能量守恒定律得到
				v质量和 = v质量i + v质量j
				v质量差 = v质量i - v质量j
				v变化[i].m速度 = v速度i * (v质量差 / v质量和) + v速度j * (v质量j * 2 / v质量和) - v速度i
				v变化[j].m速度 = v速度i * (v质量i * 2 / v质量和) + v速度j * (-v质量差 / v质量和) - v速度j
	#应用变化
	for i in v物体:
		v物体[i].m位置 += v变化[i].m位置
		v物体[i].m速度 += v变化[i].m速度
		v圆心[i][0] = v物体[i].m位置
init()
v窗口 = ti.ui.Window("碰撞1d", res = c窗口半径)
v画布 = v窗口.get_canvas()
v上次时间 = time.time()
while v窗口.running:
	#时间
	if (v这次时间 := time.time()) > v上次时间:
		v时间差 = v这次时间 - v上次时间
		if v时间差 > c最大时间差:
			v时间差 = c最大时间差
		compute(v时间差)
		v上次时间 = v这次时间
	#绘制
	v画布.circles(v圆心, c半径, (1, 1, 1))
	v窗口.show()