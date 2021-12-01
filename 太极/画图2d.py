import taichi as ti
ti.init(arch = ti.gpu)
n = 800
p = ti.Vector.field(2, dtype = float, shape = 2)
v光标位置 = ti.Vector.field(2, dtype = float, shape = 2)
@ti.kernel
def init():
	p[1][0] = 1
	p[1][1] = 1
v窗口 = ti.ui.Window("画图", res = (n, n))
v画布 = v窗口.get_canvas()
init()
while v窗口.running:
	x, y = v窗口.get_cursor_pos()
	v画布.circles(p, radius = 0.1, color = (1, 1, 1))
	#计算方向,再从方向计算位置
	v方向 = ti.atan2(x - 0.5, y - 0.5)
	v亮度 = v方向 / 6.2831853
	v光标位置[1][0] = x
	v光标位置[1][1] = y
	v画布.lines(v光标位置, width = 0.01, color = (x, y, v亮度))
	v窗口.show()
