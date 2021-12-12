import taichi as ti
import time
ti.init(arch = ti.gpu)
PI = 3.14159265
t向量2 = ti.types.vector(2, float)
c窗口宽度 = 800
c窗口高度 = 800
c窗口尺寸 = (c窗口宽度, c窗口高度)
c最大时间差 = 0.01
#物理参数
g = 1.0	#重力加速度
c粒子半径 = 0.002
c粒子直径 = c粒子半径 * 2
c支撑半径 = c粒子半径 * 4
c粒子黏度 = 0.001
c粒子压强 = 0.005
c速度损失 = 0.9
t粒子 = ti.types.struct(
	m位置 = t向量2,
	m速度 = t向量2,
	m压力 = t向量2,
	m粘力 = t向量2,
	m邻居数 = int,
	m密度 = float,
	m压强 = float
)
c重力 = t向量2(0.0, -g)
c生成宽度 = 100	#生成粒子区域的宽和高
c生成高度 = 100
c粒子数量 = c生成宽度 * c生成高度
c邻居数量 = min(c粒子数量, 1000)
v粒子 = t粒子.field(shape = c粒子数量)
v邻居索引 = ti.field(int, shape = (c粒子数量, c邻居数量))
v圆心 = t向量2.field(shape = c粒子数量)
c矩形 = (0.7, 1, 0.95, 0.05)	#左上右下
v矩形线条顶点 = t向量2.field(shape = 6)
#函数
@ti.func
def border_collision(a位置: float, a速度: float):
	v位置 = a位置
	v速度 = a速度
	if v位置 < 0:
		v位置 = 0
		v速度 = -v速度 * c速度损失
	elif v位置 > 1:
		v位置 = 1
		v速度 = -v速度 * c速度损失
	return v位置, v速度
@ti.func
def border_density(a位置: float):
	x = 0.0
	if a位置 < 0 + c粒子半径:	#靠近墙壁时x达到最大,为粒子半径
		x = c粒子半径 - a位置
	elif a位置 > 1 - c粒子半径:
		x = a位置 - 1 + c粒子半径
	x = x / c粒子半径 * 2
	return x
@ti.func
def rectangle_collision(a位置, a速度):
	v位置 = a位置
	v速度 = a速度
	if c矩形[0] < a位置.x < c矩形[2] and c矩形[3] < a位置.y:
		#靠近哪条边
		v左 = a位置.x - c矩形[0]
		v右 = c矩形[2] - a位置.x
		v下 = a位置.y - c矩形[3]
		if v左 < v右:
			if v左 < v下:	#左
				v位置.x = c矩形[0]
				v速度.x = -ti.abs(v速度.x) * c速度损失
			else:	#下
				v位置.y = c矩形[3]
				v速度.y = -ti.abs(v速度.y) * c速度损失
		else:
			if v右 < v下:	#右
				v位置.x = c矩形[2]
				v速度.x = ti.abs(v速度.x) * c速度损失
			else:	#下
				v位置.y = c矩形[3]
				v速度.y = -ti.abs(v速度.y) * c速度损失
	return v位置, v速度
@ti.func
def rectangle_density(a位置):
	x = 0.0
	v左 = a位置.x - c矩形[0] - c粒子半径
	v右 = c矩形[2] + c粒子半径 - a位置.x
	v下 = a位置.y - c矩形[3] - c粒子半径
	if 0 < v左 < c粒子半径:
		x = v左
	elif 0 < v右 < c粒子半径:
		x = v右
	elif 0 < v下 < c粒子半径:
		x = v下
	x = x / c粒子半径 * 2
	return x
@ti.kernel
def init():
	#粒子
	v粒子间隔 = c粒子直径 * 1.5
	for i, j in ti.ndrange(c生成宽度, c生成高度):
		k = i + j * c生成宽度
		v粒子[k].m位置 = t向量2(0.01 + i * v粒子间隔, 0.99 - j * v粒子间隔)
		v粒子[k].m速度 = t向量2(0, 0)
	#矩形
	v矩形线条顶点[0] = t向量2(c矩形[0], c矩形[1])
	v矩形线条顶点[1] = t向量2(c矩形[0], c矩形[3])
	v矩形线条顶点[2] = t向量2(c矩形[0], c矩形[3])
	v矩形线条顶点[3] = t向量2(c矩形[2], c矩形[3])
	v矩形线条顶点[4] = t向量2(c矩形[2], c矩形[1])
	v矩形线条顶点[5] = t向量2(c矩形[2], c矩形[3])
@ti.kernel
def compute(dt: float):
	#清除状态
	for i in v粒子:
		v粒子[i].m邻居数 = 0
		v粒子[i].m密度 = 1.0
		v粒子[i].m压力 = t向量2(0, 0)
		v粒子[i].m粘力 = t向量2(0, 0)
	#邻居,密度
	for i in v粒子:	#不能同时读写粒子i和j,有数据竞争
		for j in range(c粒子数量):
			v距离 = (v粒子[i].m位置 - v粒子[j].m位置).norm()
			if v距离 < c支撑半径:
				#邻居
				v邻居索引[i, v粒子[i].m邻居数] = j
				v粒子[i].m邻居数 += 1
	#密度
	for i in v粒子:
		#两个粒子挨得越近,密度越大
		for j0 in range(v粒子[i].m邻居数):
			j = v邻居索引[i, j0]
			v距离 = (v粒子[i].m位置 - v粒子[j].m位置).norm()
			q = v距离 / c支撑半径
			if q <= 0.5:	#计算平滑曲线
				q = (6.0 * q**3 - 6.0 * q**2 + 1)
			else:
				q = 2 * (1 - q)**3
			v粒子[i].m密度 += q
		#粒子越靠近边界,密度越大
		v粒子[i].m密度 += border_density(v粒子[i].m位置.x) + border_density(v粒子[i].m位置.y)
		v粒子[i].m密度 += rectangle_density(v粒子[i].m位置)
	#粘力,被周围粒子带动的力
	for i in v粒子:
		for j0 in range(v粒子[i].m邻居数):
			j = v邻居索引[i, j0]
			v位置差 = v粒子[j].m位置 - v粒子[i].m位置
			v距离 = v位置差.norm()
			v方向 = v位置差.normalized()
			if v距离 > 0:
				v粒子[i].m粘力 += c粒子黏度 * (v粒子[j].m速度 - v粒子[i].m速度).dot(v方向) / v粒子[j].m密度 * v方向 / (v距离 + 0.1*c支撑半径) / dt
	#压力,密度高的区域往密度低的区域移动
	for i in v粒子:
		for j0 in range(v粒子[i].m邻居数):
			j = v邻居索引[i, j0]
			v位置差 = v粒子[j].m位置 - v粒子[i].m位置
			v距离 = v位置差.norm()
			if v距离 > 0:
				v方向 = v位置差.normalized()
				v粒子[i].m压力 -= c粒子压强 * v方向 * v粒子[j].m密度 / v距离
	#位置
	for i in v粒子:
		v合力 = c重力 + v粒子[i].m粘力 + v粒子[i].m压力
		v粒子[i].m速度 += v合力 * dt
		v粒子[i].m位置 += v粒子[i].m速度 * dt
		v粒子[i].m位置.x, v粒子[i].m速度.x = border_collision(v粒子[i].m位置.x, v粒子[i].m速度.x)
		v粒子[i].m位置.y, v粒子[i].m速度.y = border_collision(v粒子[i].m位置.y, v粒子[i].m速度.y)
		v粒子[i].m位置, v粒子[i].m速度 = rectangle_collision(v粒子[i].m位置, v粒子[i].m速度)
@ti.kernel
def paint():
	for i in range(c粒子数量):
		v圆心[i] = v粒子[i].m位置
#窗口
v窗口 = ti.ui.Window("连通器", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
v上次时间 = time.time()
while v窗口.running:
	#时间
	if (v这次时间 := time.time()) > v上次时间:
		v时间差 = v这次时间 - v上次时间
		if v时间差 > c最大时间差:
			v时间差 = c最大时间差
		compute(v时间差)
		paint()
		v上次时间 = v这次时间
	#绘制
	v画布.circles(v圆心, c粒子半径, color = (.9, .9, 1))
	v画布.lines(v矩形线条顶点, 0.01, color = (1, 1, 1))
	v窗口.show()
