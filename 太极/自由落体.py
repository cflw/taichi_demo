import taichi as ti
ti.init(arch = ti.gpu)
c数量 = 50
c半径 = 10
g = 1	#重力加速度
dt = 1 / 60.0
v点 = ti.Vector.field(2, float, c数量)
v速度 = ti.Vector.field(2, float, c数量)
@ti.kernel
def init():
	for i in v点:
		v点[i] = ti.Vector([ti.random(), ti.random()])
		v速度[i] = ti.Vector([(ti.random() - 0.5), ti.random()])
@ti.kernel
def compute():
	for i in v点:
		v速度[i][1] -= g * dt
		v点[i] += v速度[i] * dt
		if v点[i][1] < 0:	#底部
			v点[i][1] = 0
			v速度[i][1] = -v速度[i][1]
			v速度[i] *= 0.9	#速度损失
		if v点[i][0] < 0:	#左边
			v点[i][0] = 0
			v速度[i][0] = -v速度[i][0]
			v速度[i] *= 0.9
		elif v点[i][0] > 1:	#右边
			v点[i][0] = 1
			v速度[i][0] = -v速度[i][0]
			v速度[i] *= 0.9
init()
v窗口 = ti.GUI("自由落体", res = (800, 800))
while v窗口.running:
	compute()
	v窗口.circles(v点.to_numpy(), color = 0xffffff, radius = c半径)
	v窗口.show()