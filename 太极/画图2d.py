import taichi as ti
import cflw代码库py.cflw时间 as 时间
ti.init(arch = ti.gpu)
n = 800
p = ti.Vector.field(2, dtype = float, shape = 2)
v光标位置 = ti.Vector.field(2, dtype = float, shape = 2)
@ti.kernel
def init():
	p[1][0] = 1
	p[1][1] = 1
v窗口 = ti.ui.Window("color", res = (n, n), vsync = True)
init()
v计时器 = 时间.C计时器(1 / 60)
while v窗口.running:
	if not v计时器.f滴答():
		continue
	x, y = v窗口.get_cursor_pos()
	v画布 = v窗口.get_canvas()
	v画布.circles(p, radius = 0.1, color = (1, 1, 1))
	#计算方向,再从方向计算位置
	v方向 = ti.atan2(x - 0.5, y - 0.5)
	v亮度 = v方向 / 6.2831853
	v光标位置[1][0] = ti.sin(v方向) * 0.5 + 0.5
	v光标位置[1][1] = ti.cos(v方向) * 0.5 + 0.5
	v画布.circles(v光标位置, radius = 0.01, color = (x, y, v亮度))
	v窗口.show()
