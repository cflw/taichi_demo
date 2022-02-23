import taichi as ti
from 公共 import *
ti.init(arch = ti.gpu)
#画布
c宽度 = 400
c高度 = 400
c尺寸 = (c宽度, c高度)
c宽高比 = c宽度 / c高度
v缓冲 = t向量3.field(shape = c尺寸)
v图像 = t向量3.field(shape = c尺寸)
c墙壁半径 = 100
#渲染参数
c采样数 = 4	#每像素光线数
c最大深度 = 20
c表面采样 = True
c继续概率 = 0.9
c模糊率 = 0.4
#场景,使用右手坐标
v场景 = C场景()
v场景.add(C球体(t向量3(0, 5.4, -1), 3, t向量3(10, 10, 10), E材质.e光源))	#光
v场景.add(C球体(t向量3(c墙壁半径 + 1.5, 0, -1), c墙壁半径, t向量3(.8, .3, .3), E材质.e漫反射))	#镜头左
v场景.add(C球体(t向量3(-c墙壁半径 - 1.5, 0, -1), c墙壁半径, t向量3(.3, .3, .8), E材质.e漫反射))	#镜头右
v场景.add(C球体(t向量3(0, 1, c墙壁半径 + 1), c墙壁半径, t向量3(.8, .8, .8), E材质.e漫反射))	#镜头前面的墙
v场景.add(C球体(t向量3(0, 1, -c墙壁半径 - 6), c墙壁半径, t向量3(.2, .2, .2), E材质.e漫反射))#镜头背后的墙
v场景.add(C球体(t向量3(0, c墙壁半径 + 2.5, -1), c墙壁半径, t向量3(.3, .8, .3), E材质.e漫反射))	#镜头上
v场景.add(C球体(t向量3(0, -c墙壁半径 - 0.5, -1), c墙壁半径, t向量3(.8, .8, .8), E材质.e漫反射))	#镜头下
v场景.add(C球体(t向量3(-0.8, 0.2, -1), 0.7, t向量3(.9, .9, .9), E材质.e镜面反射))	#镜面球
v场景.add(C球体(t向量3(0.7, 0, -0.5), 0.5, t向量3(1, 1, 1), E材质.e折射))	#玻璃球
v场景.add(C球体(t向量3(0, -0.2, -1.5), 0.3, t向量3(0.8, 0.3, 0.3), E材质.e漫反射))	#漫反射球
v场景.add(C球体(t向量3(0.6, -0.3, -2.0), 0.2, t向量3(0.8, 0.6, 0.2), E材质.e模糊反射))	#模糊金属球
v相机 = C相机(t向量3(0.0, 1.0, -5.0), t向量3(0.0, 1.0, -1.0), t向量3(0.0, 1.0, 0.0))
v透视 = C透视投影(c宽高比, PI / 3)
v取景框 = C取景框(v相机, v透视)
#渲染
@ti.kernel
def paint(frame: float):
	for i, j in v缓冲:
		v颜色 = t向量3(0, 0, 0)
		for k in range(c采样数):
			u = (i + ti.random()) / c宽度
			v = (j + ti.random()) / c高度
			v光线 = v取景框.get_ray(u, v)
			v颜色 += ray_color(v光线)
		v缓冲[i, j] += v颜色
		v图像[i, j] = v缓冲[i, j] / (frame * c采样数)
@ti.func
def ray_color1(a光线):	#只有物体颜色
	v光线位置 = a光线.m位置
	v光线方向 = a光线.m方向
	v光线颜色 = t向量3(1, 1, 1)
	v碰撞, v交点, v交点法线, v前面, v物体颜色, v材质 = v场景.hit(C光线(v光线位置, v光线方向))
	if v碰撞:
		v光线颜色 = v物体颜色
	return v光线颜色
@ti.func
def ray_color2(a光线):	#朗伯反射
	v光线位置 = a光线.m位置
	v光线方向 = a光线.m方向
	v光线颜色 = t向量3(1, 1, 1)
	v碰撞, v交点, v交点法线, v前面, v物体颜色, v材质 = v场景.hit(C光线(v光线位置, v光线方向))
	if v碰撞:
		if v材质 == E材质.e光源:
			v光线颜色 = v物体颜色
		else:
			v到光源 = (t向量3(0, 5.4 - 3, -1) - v交点).normalized()
			v光线颜色 = v物体颜色 * ti.max(v到光源.dot(v交点法线) / v到光源.norm() * v交点法线.norm(), 0.0)
	return v光线颜色
@ti.func
def ray_color(a光线):	#光线追踪
	v当前深度 = 0
	v光线位置 = a光线.m位置
	v光线方向 = a光线.m方向
	v光线颜色 = t向量3(1, 1, 1)
	v碰撞, v交点, v交点法线, v前面, v物体颜色, v材质 = v场景.hit(C光线(v光线位置, v光线方向))
	while v碰撞:
		#结束条件
		v当前深度 += 1
		if v当前深度 > 5 and ti.random() > c继续概率:
			v光线颜色 /= c继续概率
			break
		elif v当前深度 > c最大深度:
			break
		#根据材质决定如何反射光线
		if v材质 == E材质.e光源:
			v光线颜色 *= v物体颜色
			break
		elif v材质 == E材质.e漫反射:
			v目标 = v交点 + v交点法线
			if c表面采样:
				v目标 += random_unit_vector()
			else:
				v目标 += random_in_unit_sphere()
			v光线方向 = v目标 - v交点
			v光线位置 = v交点
			v光线颜色 *= v物体颜色
		elif v材质 == E材质.e镜面反射:
			v光线方向 = reflect(v光线方向.normalized(), v交点法线)
			if v光线方向.dot(v交点法线) < 0:
				break
			v光线位置 = v交点
			v光线颜色 *= v物体颜色
		elif v材质 == E材质.e折射:
			v折射率 = 1.5
			if v前面:
				v折射率 = 1 / v折射率
			cosθ = min(-v光线方向.normalized().dot(v交点法线), 1.0)
			sinθ = ti.sqrt(1 - cosθ * cosθ)
			if v折射率 * sinθ > 1 or reflectance(cosθ, v折射率) > ti.random():	#全反射
				v光线方向 = reflect(v光线方向.normalized(), v交点法线)
			else:	#折射
				v光线方向 = refract(v光线方向.normalized(), v交点法线, v折射率)
			v光线位置 = v交点
			v光线颜色 *= v物体颜色
		elif v材质 == E材质.e模糊反射:
			v光线方向 = reflect(v光线方向.normalized(), v交点法线)
			if c表面采样:
				v光线方向 += random_unit_vector() * c模糊率
			else:
				v光线方向 += random_in_unit_sphere() * c模糊率
			if v光线方向.dot(v交点法线) < 0:
				break
			v光线位置 = v交点
			v光线颜色 *= v物体颜色
		#新的循环
		v碰撞, v交点, v交点法线, v前面, v物体颜色, v材质 = v场景.hit(C光线(v光线位置, v光线方向))
	return v光线颜色
#窗口
v窗口 = ti.ui.Window("康奈尔盒", res = c尺寸)
v画布 = v窗口.get_canvas()
v帧 = 0.0
while v窗口.running:
	v帧 += 1
	paint(v帧)
	v画布.set_image(v图像)
	v窗口.show()