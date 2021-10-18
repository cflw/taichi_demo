import taichi as ti
import time
ti.init(arch = ti.gpu)
c窗口边长 = 800
c窗口尺寸 = (c窗口边长, c窗口边长)
c时间间隔 = 1 / 100
c半径 = 0.01
c直径 = c半径 * 2
N = 10	#数量
G = 1	#引力常量
v质量 = ti.field(dtype = float, shape = N)
v位置 = ti.Vector.field(2, dtype = float, shape = N)
v速度 = ti.Vector.field(2, dtype = float, shape = N)
v力 = ti.Vector.field(2, dtype = float, shape = N)
@ti.kernel
def init():
	for i in ti.static(range(N)):
		v质量[i] = 1
		v位置[i][0] = ti.random()
		v位置[i][1] = ti.random()
		v速度[i][0] = v位置[i][1] * 0.5
		v速度[i][1] = -v位置[i][0] * 0.5
@ti.kernel
def compute():
	for i in ti.static(range(N)):
		v力[i] = [0, 0]
	for i in ti.static(range(N)):
		for j in ti.static(range(N)):
			if i != j:
				diff = v位置[i] - v位置[j]
				r = diff.norm()
				f = -G * v质量[i] * v质量[j] / r * diff
				v力[i] += f
	for i in ti.static(range(N)):
		v速度[i] += v力[i] / v质量[i] * c时间间隔
		v位置[i] += v速度[i] * c时间间隔
v窗口 = ti.ui.Window("万有引力2d", res = c窗口尺寸)
init()
while v窗口.running:
	compute()
	v画布 = v窗口.get_canvas()
	v画布.circles(v位置, radius = c半径, color = (1, 1, 1))
	v窗口.show()
	time.sleep(c时间间隔)