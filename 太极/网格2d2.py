import taichi as ti
ti.init(arch = ti.gpu)
t向量2 = ti.types.vector(2, float)
#窗口
c窗口宽度 = 800
c窗口高度 = 800
c窗口尺寸 = (c窗口宽度, c窗口高度)
#网格
c网格宽度 = 16
c网格高度 = 16
v点顶点 = t向量2.field(shape = (c网格宽度, c网格高度))
c横线数 = c网格高度 * (c网格宽度 - 1)
c竖线数 = c网格宽度 * (c网格高度 - 1)
c对角线数 = (c网格宽度 - 1) * (c网格高度 - 1)
c网格线条数 = c横线数 + c竖线数 + c对角线数	#总线条数
t线条 = ti.types.struct(	#表示线条两个点在点顶点中的索引
	x0 = int,
	y0 = int,
	x1 = int,
	y1 = int,
)
v线条 = t线条.field(shape = c网格线条数)
v线顶点 = t向量2.field(shape = c网格线条数 * 2)
@ti.kernel
def init():
	#给点赋值
	for i, j in v点顶点:
		v点顶点[i, j] = ti.Vector([i / c网格宽度, j / c网格高度])
	#给线条赋值
	for i, j in ti.ndrange(c网格宽度 - 1, c网格高度):	#横线
		k = i + j * (c网格宽度 - 1)
		v线条[k] = t线条(x0 = i, y0 = j, x1 = i+1, y1 = j)
	for i, j in ti.ndrange(c网格宽度, c网格高度 - 1):	#竖线
		k = c横线数 + j + i * (c网格高度 - 1)
		v线条[k] = t线条(x0 = i, y0 = j, x1 = i, y1 = j+1)
	for i, j in ti.ndrange(c网格宽度 - 1, c网格高度 - 1):	#对角线
		k = c横线数 + c竖线数 + i + j * (c网格宽度 - 1)
		v线条[k] = t线条(x0 = i, y0 = j, x1 = i+1, y1 = j+1)
	#给线顶点赋值
	for i in v线条:
		k = i * 2
		v当前线 = v线条[i]
		v线顶点[k] = v点顶点[v当前线.x0, v当前线.y0]
		v线顶点[k+1] = v点顶点[v当前线.x1, v当前线.y1]
#窗口
v窗口 = ti.ui.Window("网格2d", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
while v窗口.running:
	v画布.lines(v线顶点, 0.01, color = (1, 1, 1))	#线列表,每2个顶点组成一条线
	v窗口.show()