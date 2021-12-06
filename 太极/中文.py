#测试太极程序存在中文字符时的运行情况
import taichi as ti
ti.init(arch = ti.gpu)
常量 = 1
变量 = ti.field(float, ())
@ti.func
def 函数():
	return 1
@ti.kernel
def 内核():
	print(函数())
内核()