import taichi as ti
v窗口 = ti.GUI("鼠标坐标", res = (1000, 1000))
v窗口.limit_fps = 10
while v窗口.running:
	x, y = v窗口.get_cursor_pos()
	print(x, y)
	v窗口.show()