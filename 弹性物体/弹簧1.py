import taichi as ti
import time
ti.init(arch = ti.gpu)
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
#窗口
c宽度 = 800
c高度 = 800
c尺寸 = (c宽度, c高度)
#小球
c半径 = 0.05
v位置 = ti.Vector.field(2, float, shape = 1)
v速度 = ti.Vector.field(2, float, shape = 1)
v顶点 = ti.Vector.field(2, float, shape = 2)
#物理参数
K = 20	#劲度系数
m = 1	#物体质量
g = 9.8	#重力加速度
X = 0
c顶端 = t向量2(0.5, 1.0)
#变量
#初始化
@ti.kernel
def init():
	v位置[0] = ti.Vector([ti.random(), ti.random()])
	v速度[0] = ti.Vector([0, 0])
	v顶点[0] = v位置[0]
	v顶点[1] = c顶端
#计算
@ti.kernel
def compute(dt: float):
	v位置差 = c顶端 - v位置[0]
	v方向 = v位置差.normalized()
	v距离 = v位置差.norm()
	v力 = K * (v距离 - X)
	v加速度 = v力 / m
	v速度[0] += v加速度 * dt * v方向
	v速度[0].y -= g * dt
	v速度[0] -= v速度[0] * dt * 0.3	#速度损失
	v位置[0] += v速度[0] * dt
	v顶点[0] = v位置[0]
#窗口
v窗口 = ti.ui.Window("弹簧1", res = c尺寸)
v画布 = v窗口.get_canvas()
init()
compute(0.001)
v上次时间 = time.time()
while v窗口.running:
	if (v这次时间 := time.time()) > 0:
		v时间差 = v这次时间 - v上次时间
		compute(v时间差)
		v上次时间 = v这次时间
	v画布.set_background_color((0, 0, 0))
	v画布.lines(v顶点, 0.01, color = (0.5, 0.5, 0.5))
	v画布.circles(v位置, c半径, (1, 1, 1))
	v窗口.show()