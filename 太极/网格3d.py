import taichi as ti
import time
ti.init(arch = ti.gpu)
PI = 3.14159265
半PI = 1.5707963
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
t向量4 = ti.types.vector(4, float)
t矩阵4 = ti.types.matrix(4, 4, float)
#===============================================================================
# 数据
#===============================================================================
#窗口
c窗口宽度 = 800
c窗口高度 = 800
c窗口尺寸 = (c窗口宽度, c窗口高度)
c宽高比 = c窗口宽度 / c窗口高度
#网格
c网格长度 = 4	#每一条边的顶点数
c网格宽度 = 4
c网格高度 = 4
c网格前面线顶点数 = c网格长度 * c网格高度 * 2
c网格侧面线顶点数 = c网格宽度 * c网格高度 * 2
c网格顶面线顶点数 = c网格长度 * c网格宽度 * 2
c网格线条数 = c网格前面线顶点数 + c网格侧面线顶点数 + c网格顶面线顶点数
v线顶点0 = t向量3.field(shape = c网格线条数)	#原始顶点
v线顶点1 = t向量2.field(shape = c网格线条数)	#显示顶点
c边界0 = -0.2
c边界1 = 0.2
#变换
v世界 = t矩阵4.field(shape = ())
v视图 = t矩阵4.field(shape = ())
v投影 = t矩阵4.field(shape = ())
#===============================================================================
# 计算
#===============================================================================
#矩阵
@ti.func
def view_matrix(e, a, u):	#视图矩阵,眼 在 上
	z = (e - a).normalized()
	x = u.cross(z).normalized()
	y = z.cross(x)
	return t矩阵4([
		[x.x,	x.y,	x.z,	-x.dot(e)],
		[y.x,	y.y,	y.z,	-y.dot(e)],
		[z.x,	z.y,	z.z,	-z.dot(e)],
		[0,		0,		0,		1]
	])
@ti.func
def projection_matrix(f, a, zn, zf):	#投影矩阵,视角 宽高比 近 远
	y = 1.0 / ti.tan(f * 0.5)
	x = y / a
	q1 = zn / (zf - zn)
	q2 = zf * q1
	return t矩阵4([
		[x,		0,		0,		0],
		[0,		y,		0,		0],
		[0,		0,		q1,		q2],
		[0,		0,		-1,		0]
	])
#通用
@ti.func
def lerp(a: float, b: float, t: float):
	return a + (b - a) * t
@ti.func
def position(i: float, size: float):
	return lerp(c边界0, c边界1, i / (size - 1.0))
#内核
@ti.kernel
def init():
	#给线赋值
	for x, y in ti.ndrange(c网格长度, c网格高度):
		k = (x + y * c网格长度) * 2
		x0 = position(x, c网格长度)
		y0 = position(y, c网格高度)
		v线顶点0[k] = t向量3(x0, y0, c边界0)
		v线顶点0[k + 1] = t向量3(x0, y0, c边界1)
	for z, y in ti.ndrange(c网格宽度, c网格高度):
		k = c网格前面线顶点数 + (z + y * c网格宽度) * 2
		z0 = position(z, c网格长度)
		y0 = position(y, c网格高度)
		v线顶点0[k] = t向量3(c边界0, y0, z0)
		v线顶点0[k + 1] = t向量3(c边界1, y0, z0)
	for x, z in ti.ndrange(c网格长度, c网格宽度):
		k = c网格前面线顶点数 + c网格侧面线顶点数 + (x + z * c网格长度) * 2
		x0 = position(x, c网格长度)
		z0 = position(z, c网格高度)
		v线顶点0[k] = t向量3(x0, c边界0, z0)
		v线顶点0[k + 1] = t向量3(x0, c边界1, z0)
@ti.kernel
def compute(t: float):
	v视图[None] = view_matrix(t向量3(ti.cos(t) * 10, 5, ti.sin(t) * 10), t向量3(0, 0, 0), t向量3(0, 1, 0))
	v投影[None] = projection_matrix(半PI, c宽高比, 0.0001, 10000)
@ti.kernel
def paint():
	for i in v线顶点0:
		v顶点 = t向量4(v线顶点0[i], 1)
		v顶点 = v视图[None] @ v顶点
		v顶点 = v投影[None] @ v顶点
		v线顶点1[i] = t向量2(v顶点.x + 0.5, v顶点.y + 0.5)
#===============================================================================
# 输出
#===============================================================================
v窗口 = ti.ui.Window("网格3d", res = c窗口尺寸, vsync = True)
v画布 = v窗口.get_canvas()
init()
v上次时间 = time.time()
v经过时间 = 0
while v窗口.running:
	v这次时间 = time.time()
	v经过时间 += v这次时间 - v上次时间
	v上次时间 = v这次时间
	compute(v经过时间)
	paint()
	v画布.lines(v线顶点1, 0.001, color = (1, 1, 1))
	v窗口.show()