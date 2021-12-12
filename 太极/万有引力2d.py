import taichi as ti
import time
ti.init(arch = ti.gpu)
t向量2 = ti.types.vector(2, float)
c窗口边长 = 800
c窗口尺寸 = (c窗口边长, c窗口边长)
c时间间隔 = 1 / 100
c半径 = 0.01
c直径 = c半径 * 2
c数量 = 10
G = 1	#引力常量
v质量 = ti.field(dtype = float, shape = c数量)
v位置 = t向量2.field(shape = c数量)
v速度 = t向量2.field(shape = c数量)
v力 = t向量2.field(shape = c数量)
@ti.kernel
def init():
	for i in range(c数量):
		v质量[i] = 1
		v位置[i] = t向量2(ti.random(), ti.random())
		v速度[i] = t向量2(v位置[i].y, -v位置[i].x) * 0.5
@ti.kernel
def compute():
	for i in range(c数量):
		v力[i] = t向量2(0, 0)
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
v窗口 = ti.ui.Window("万有引力2d", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
while v窗口.running:
	compute()
	v画布.circles(v位置, radius = c半径, color = (1, 1, 1))
	v窗口.show()
	time.sleep(c时间间隔)