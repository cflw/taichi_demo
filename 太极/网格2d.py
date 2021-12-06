import taichi as ti
ti.init(arch = ti.gpu)
t向量2 = ti.types.vector(2, float)
#窗口
c窗口宽度 = 800
c窗口高度 = 800
c窗口尺寸 = (c窗口宽度, c窗口高度)
#网格
c网格宽度 = 4
c网格高度 = 4
v点顶点 = t向量2.field(shape = (c网格宽度, c网格高度))
c网格线条数 = 2 * c网格宽度 * c网格高度 - c网格宽度 - c网格高度	#宽 * (高 - 1) + 高 * (宽 - 1)
v线顶点 = t向量2.field(shape = c网格线条数 * 2)	#无对角线
@ti.kernel
def init():
	#给点赋值
	for i, j in v点顶点:
		v点顶点[i, j] = ti.Vector([i / c网格宽度, j / c网格高度])
	#给线赋值
	for i, j in v点顶点:
		if i < c网格宽度 - 1:	#横线
			k = (i + j * (c网格宽度 - 1)) * 2
			v线顶点[k] = v点顶点[i, j]
			v线顶点[k + 1] = v点顶点[i + 1, j]
		if j < c网格高度 - 1:	#竖线
			k = c网格线条数 + (j + i * (c网格高度 - 1)) * 2
			v线顶点[k] = v点顶点[i, j]
			v线顶点[k + 1] = v点顶点[i, j + 1]
	#调试打印
	# for i in v线顶点:
	# 	print(v线顶点[i])
#窗口
v窗口 = ti.ui.Window("网格2d", res = c窗口尺寸)
v画布 = v窗口.get_canvas()
init()
while v窗口.running:
	v画布.lines(v线顶点, 0.01, color = (1, 1, 1))	#线列表,每2个顶点组成一条线
	v窗口.show()