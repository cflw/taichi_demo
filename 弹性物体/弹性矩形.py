import taichi as ti
import time
ti.init(arch = ti.gpu)
#===============================================================================
# 数据
#===============================================================================
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
#窗口
c窗口宽度 = 800
c窗口高度 = 800
c窗口尺寸 = (c窗口宽度, c窗口高度)
#物体
c网格宽度 = 4
c网格高度 = 4
c横线数 = c网格高度 * (c网格宽度 - 1)
c竖线数 = c网格宽度 * (c网格高度 - 1)
c对角线数 = (c网格宽度 - 1) * (c网格高度 - 1)
c网格线条数 = c横线数 + c竖线数 + c对角线数	#总线条数
t物质点 = ti.types.struct(
	m位置 = t向量2,
	m速度 = t向量2,
)
v点 = t物质点.field(shape = (c网格宽度, c网格高度))
t弹簧 = ti.types.struct(
	x0 = int,
	y0 = int,
	x1 = int,
	y1 = int,
	m初始长度 = float	#弹力为零 时的长度
)
v线 = t弹簧.field(shape = c网格线条数)
v线顶点 = t向量2.field(shape = c网格线条数 * 2)
#物理参数
K = 10000	#劲度系数
m = 1	#点质量
g = 0.5	#重力加速度
#===============================================================================
# 计算
#===============================================================================
@ti.func
def new_line(x0: int, y0: int, x1: int, y1: int):
	v长度 = (v点[x0, y0].m位置 - v点[x1, y1].m位置).norm()
	return t弹簧(x0 = x0, y0 = y0, x1 = x1, y1 = y1, m初始长度 = v长度)
@ti.func
def compute_change_velocity(p0, p1, X: float, dt: float):	#根据两点位置计算出点0的速度变化量
	v位置差 = p1 - p0
	v距离 = v位置差.norm()
	v速度变化 = t向量2(0, 0)
	if v距离 != X:
		v方向 = v位置差.normalized()
		v长度差 = v距离 - X	#正数表示拉长了
		v力 = K * v长度差
		v加速度 = v力 / m
		v速度变化 = v加速度 * dt * v方向
	return v速度变化
@ti.kernel
def init():
	#点
	for i, j in v点:
		v点[i, j].m位置 = ti.Vector([i / c网格宽度 * 0.4 + 0.4, j / c网格高度 * 0.4 + 0.6])
		v点[i, j].m速度 = t向量2(0, 0)
	#线(弹簧)
	for i, j in ti.ndrange(c网格宽度 - 1, c网格高度):	#横线
		k = i + j * (c网格宽度 - 1)
		v线[k] = new_line(i, j, i+1, j)
	for i, j in ti.ndrange(c网格宽度, c网格高度 - 1):	#竖线
		k = c横线数 + j + i * (c网格高度 - 1)
		v线[k] = new_line(i, j, i, j+1)
	for i, j in ti.ndrange(c网格宽度 - 1, c网格高度 - 1):	#对角线
		k = c横线数 + c竖线数 + i + j * (c网格宽度 - 1)
		v线[k] = new_line(i, j, i+1, j+1)
@ti.kernel
def compute(dt: float):
	for i in v线:	#弹簧
		v线0 = v线[i]
		v点0 = v点[v线0.x0, v线0.y0]
		v点1 = v点[v线0.x1, v线0.y1]
		v速度变化0 = compute_change_velocity(v点0.m位置, v点1.m位置, v线0.m初始长度, dt)
		#隐式欧拉法，计算下一时刻的状态
		v新位置0 = v点0.m位置 + (v点0.m速度 + v速度变化0) * dt
		v新位置1 = v点1.m位置 + (v点1.m速度 - v速度变化0) * dt
		v速度变化1 = compute_change_velocity(v新位置0, v新位置1, v线0.m初始长度, dt)
		v点[v线0.x0, v线0.y0].m速度 += v速度变化1
		v点[v线0.x1, v线0.y1].m速度 -= v速度变化1
	for i, j in v点:	#运动,边框反弹
		v点[i,j].m速度.y -= g * dt
		v点[i,j].m位置 += v点[i,j].m速度 * dt
		if v点[i,j].m位置.x < 0:	#边界反弹
			v点[i,j].m位置.x = 0
			v点[i,j].m速度.x = -v点[i,j].m速度.x * 0.9
		elif v点[i,j].m位置.x > 1:
			v点[i,j].m位置.x = 1
			v点[i,j].m速度.x = -v点[i,j].m速度.x * 0.9
		if v点[i,j].m位置.y < 0:
			v点[i,j].m位置.y = 0
			v点[i,j].m速度.y = -v点[i,j].m速度.y * 0.9
		elif v点[i,j].m位置.y > 1:
			v点[i,j].m位置.y = 1
			v点[i,j].m速度.y = -v点[i,j].m速度.y * 0.9
@ti.kernel
def paint():
	for i in v线:
		k = i * 2
		v当前线 = v线[i]
		v线顶点[k] = v点[v当前线.x0, v当前线.y0].m位置
		v线顶点[k+1] = v点[v当前线.x1, v当前线.y1].m位置
#===============================================================================
# 输出
#===============================================================================
v窗口 = ti.ui.Window("弹性矩形", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
compute(0.001)
paint()
v上次时间 = time.time()
while v窗口.running:
	if (v这次时间 := time.time()) > v上次时间:
		v时间差 = v这次时间 - v上次时间
		compute(v时间差)
		paint()
		v上次时间 = v这次时间
	v画布.lines(v线顶点, 0.001, color = (1, 1, 1))
	v窗口.show()