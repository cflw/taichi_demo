import taichi as ti
import time
ti.init(arch = ti.gpu)
t向量3 = ti.types.vector(3, float)
c窗口边长 = 800
c窗口尺寸 = (c窗口边长, c窗口边长)
c时间间隔 = 1 / 100
#物理
c半径 = 0.01
c直径 = c半径 * 2
c数量 = 10
G = 1	#引力常量
v质量 = ti.field(dtype = float, shape = c数量)
v位置 = t向量3.field(shape = c数量)
v速度 = t向量3.field(shape = c数量)
v力 = t向量3.field(shape = c数量)
@ti.func
def random0():
	return ti.random() - 0.5
@ti.kernel
def init():
	for i in range(c数量):
		v质量[i] = 1
		v位置[i] = t向量3(random0(), random0(), random0())
		# v速度[i][0] = random0()
		# v速度[i][1] = random0()
		# v速度[i][2] = random0()
@ti.kernel
def compute():
	for i in range(c数量):
		v力[i] = t向量3(0, 0, 0)
	for i in range(c数量):
		for j in range(c数量):
			if i != j:
				diff = v位置[i] - v位置[j]
				r = diff.norm()
				f = -G * v质量[i] * v质量[j] / r * diff
				v力[i] += f
	for i in range(c数量):
		v速度[i] += v力[i] / v质量[i] * c时间间隔
		v位置[i] += v速度[i] * c时间间隔
v窗口 = ti.ui.Window("万有引力3d", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
v场景 = ti.ui.Scene()
v相机 = ti.ui.make_camera()
v相机.position(0, 0, 2)
v相机.lookat(0, 0, 0)
v相机.up(0, 1, 0)
init()
while v窗口.running:
	compute()
	v场景.set_camera(v相机)
	v场景.point_light(pos = (0, 5, 1), color = (1, 1, 1))
	v场景.particles(v位置, radius = c半径, color = (1, 1, 1))
	v画布.scene(v场景)
	v窗口.show()
	time.sleep(c时间间隔)