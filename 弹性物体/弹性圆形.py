import taichi as ti
import time
ti.init(arch = ti.gpu)
#===============================================================================
# 数据
#===============================================================================
PI = 3.14159265
半PI = 1.5707963
二PI = 6.2831853
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
#窗口
c窗口宽度 = 800
c窗口高度 = 800
c窗口尺寸 = (c窗口宽度, c窗口高度)
#物体
c圆形半径 = 0.4
c圆形边数 = 16	#最外围有多少条边
c圆形点数 = (3 + c圆形边数) * (c圆形边数 - 2) // 2	#3 + ... + N
c圆形线条数 = 3 * c圆形点数 - c圆形边数 - 3	#总线条数,每一层线条数等于点数,每两层之间线条数为两层点数之和
t物质点 = ti.types.struct(
	m位置 = t向量2,
	m速度 = t向量2,
)
v点 = t物质点.field(shape = c圆形点数)
t弹簧 = ti.types.struct(
	a = int,
	b = int,
	m初始长度 = float	#弹力为零 时的长度
)
v线 = t弹簧.field(shape = c圆形线条数)
v线顶点 = t向量2.field(shape = c圆形线条数 * 2)
#物理参数
K = 10000	#劲度系数
m = 1	#点质量
g = 0.5	#重力加速度
#===============================================================================
# 计算
#===============================================================================
@ti.func
def new_line(a: int, b: int):
	v长度 = (v点[a].m位置 - v点[b].m位置).norm()
	return t弹簧(a = a, b = b, m初始长度 = v长度)
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
def init():	#生成的圆形网格像蜘蛛网
	#点
	for i in range(3, c圆形边数+1):
		v点开始 = (3 + i) * (i - 2) // 2 - i
		v半径 = c圆形半径 / (c圆形边数 - 2) * (i - 2)
		for j in range(i):
			k = v点开始 + j
			v方向 = j / i * 二PI
			v点[k].m位置 = ti.Vector([ti.sin(v方向) * v半径 + 0.5, ti.cos(v方向) * v半径 + 0.5])
			v点[k].m速度 = t向量2(0, 0)
	#线(弹簧)
	for i in range(3, c圆形边数+1):	#正多边形
		v点开始 = (3 + i) * (i - 2) // 2 - i
		for j in range(i):
			k = v点开始 + j
			if j != i - 1:
				v线[k] = new_line(k, k+1)
			else:
				v线[k] = new_line(k, v点开始)
	for i in range(3, c圆形边数):	#两层之间的线条
		v线开始 = c圆形点数 + (8 + 2 * i) * (i - 2) // 2 - (1 + 2 * i)	#7 + 9 + ... + (2n - 1)
		v内点开始 = (3 + i) * (i - 2) // 2 - i
		v外点开始 = v内点开始 + i
		for j in range(i):
			k = v线开始 + j * 2
			#内层的 点i 和外层的 点i,点i+1 连接
			v线[k] = new_line(v内点开始+j, v外点开始+j)
			v线[k+1] = new_line(v内点开始+j, v外点开始+j+1)
		#最后,内层的 点i 和外层的 点-1 连接
		v线[v线开始+2*i] = new_line(v内点开始, v外点开始+i)
@ti.kernel
def compute(dt: float):
	for i in v线:	#弹簧
		v线0 = v线[i]
		v点0 = v点[v线0.a]
		v点1 = v点[v线0.b]
		v速度变化0 = compute_change_velocity(v点0.m位置, v点1.m位置, v线0.m初始长度, dt)
		#隐式欧拉法，计算下一时刻的状态
		v新位置0 = v点0.m位置 + (v点0.m速度 + v速度变化0) * dt
		v新位置1 = v点1.m位置 + (v点1.m速度 - v速度变化0) * dt
		v速度变化1 = compute_change_velocity(v新位置0, v新位置1, v线0.m初始长度, dt)
		v点[v线0.a].m速度 += v速度变化1
		v点[v线0.b].m速度 -= v速度变化1
	for i in v点:	#运动,边框反弹
		v点[i].m速度.y -= g * dt
		v点[i].m位置 += v点[i].m速度 * dt
		if v点[i].m位置.x < 0:	#边界反弹
			v点[i].m位置.x = 0
			v点[i].m速度.x = -v点[i].m速度.x * 0.9
		elif v点[i].m位置.x > 1:
			v点[i].m位置.x = 1
			v点[i].m速度.x = -v点[i].m速度.x * 0.9
		if v点[i].m位置.y < 0:
			v点[i].m位置.y = 0
			v点[i].m速度.y = -v点[i].m速度.y * 0.9
		elif v点[i].m位置.y > 1:
			v点[i].m位置.y = 1
			v点[i].m速度.y = -v点[i].m速度.y * 0.9
@ti.kernel
def paint():
	for i in v线:
		k = i * 2
		v当前线 = v线[i]
		v线顶点[k] = v点[v当前线.a].m位置
		v线顶点[k+1] = v点[v当前线.b].m位置
#===============================================================================
# 输出
#===============================================================================
v窗口 = ti.ui.Window("弹性圆形", res = c窗口尺寸)
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