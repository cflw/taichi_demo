import taichi as ti
import time
ti.init(arch = ti.gpu)
PI = 3.14159265
#窗口
c窗口边长 = 800
c窗口半径 = (c窗口边长, c窗口边长)
c最大时间差 = 0.01
#物理参数
c数量 = 100
c半径 = 0.02
c直径 = c半径 * 2
c质量 = 0.1
t向量3 = ti.types.vector(3, float)
t物体 = ti.types.struct(m位置 = t向量3, m速度 = t向量3, m质量 = float)
v物体 = t物体.field(shape = c数量)
v变化 = t物体.field(shape = c数量)
v圆心 = ti.Vector.field(3, dtype = float, shape = c数量)
#计算
@ti.func
def border(a位置: float, a速度: float):
	v位置 = a位置
	v速度 = a速度
	if v位置 < -0.5:
		v位置 = -0.5
		v速度 = -v速度
	elif v位置 > 0.5:
		v位置 = 0.5
		v速度 = -v速度
	return v位置, v速度
@ti.kernel
def init():
	for i in v物体:
		v方向 = ti.random() * PI * 2
		v物体[i].m位置 = t向量3(ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5)
		v物体[i].m速度 = t向量3(ti.sin(v方向), ti.cos(v方向), ti.random()) * 0.2
		v物体[i].m质量 = c质量
@ti.kernel
def compute(dt: float):
	#位置
	for i in v物体:
		v物体[i].m位置 += v物体[i].m速度 * dt
		v物体[i].m位置.x, v物体[i].m速度.x = border(v物体[i].m位置.x, v物体[i].m速度.x)
		v物体[i].m位置.y, v物体[i].m速度.y = border(v物体[i].m位置.y, v物体[i].m速度.y)
		v物体[i].m位置.z, v物体[i].m速度.z = border(v物体[i].m位置.z, v物体[i].m速度.z)
		#清除变化
		v变化[i].m位置 = t向量3(0, 0, 0)
		v变化[i].m速度 = t向量3(0, 0, 0)
		# print(i, v物体[i].m位置, v物体[i].m速度)
	#三维小球弹性碰撞
	for i in v物体:
		v速度i = v物体[i].m速度
		v质量i = v物体[i].m质量
		for j in range(i):
			v速度j = v物体[j].m速度
			v质量j = v物体[j].m质量
			v位置差 = v物体[j].m位置 - v物体[i].m位置
			v距离 = v位置差.norm()
			if v距离 <= c直径:	#发生碰撞
				# print(i, j, v距离)
				#稍微拉开一点距离,防止下一帧再次发生碰撞
				v拉开距离 = v位置差 * (1 - v距离 / c直径)
				v变化[i].m位置 -= v拉开距离
				v变化[j].m位置 += v拉开距离
				#计算碰撞后速度,步骤和二维空间弹性斜碰一样
				v质量和 = v质量i + v质量j
				v质量差 = v质量i - v质量j
				v方向 = v位置差 / v距离
				v法线i = v方向.cross(v速度i).cross(v方向).normalized()
				v法线j = v方向.cross(v速度j).cross(v方向).normalized()
				v相对速度ix = v速度i.dot(v方向)
				v相对速度iy = v速度i.dot(v法线i)
				v相对速度jx = v速度j.dot(v方向)
				v相对速度jy = v速度j.dot(v法线j)
				v相对速度ix_ = v质量差 / v质量和 * v相对速度ix + 2 * v质量j / v质量和 * v相对速度jx
				v相对速度jx_ = -v质量差 / v质量和 * v相对速度jx + 2 * v质量i / v质量和 * v相对速度ix
				v变化[i].m速度 += v相对速度ix_*v方向 + v相对速度iy*v法线i - v速度i
				v变化[j].m速度 += v相对速度jx_*v方向 + v相对速度jy*v法线j - v速度j
	#应用变化
	for i in v物体:
		v物体[i].m位置 += v变化[i].m位置
		v物体[i].m速度 += v变化[i].m速度
		v圆心[i] = v物体[i].m位置
init()
#窗口
v窗口 = ti.ui.Window("碰撞3d", res = c窗口半径)
v画布 = v窗口.get_canvas()
v场景 = ti.ui.Scene()
v相机 = ti.ui.make_camera()
v相机.position(0, 0, 2)
v相机.lookat(0, 0, 0)
v相机.up(0, 1, 0)
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
	v场景.set_camera(v相机)
	v场景.point_light(pos = (0, 5, 1), color = (1, 1, 1))
	v场景.particles(v圆心, radius = c半径, color = (1, 1, 1))
	v画布.scene(v场景)
	v窗口.show()
