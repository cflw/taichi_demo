import taichi as ti
import enum
from PIL import Image	#pillow
import numpy as np	#numpy
#常量
PI = 3.14159265
半PI = 1.5707963
二PI = 6.2831853
c最小值 = 0.0001
c最大值 = 10e5
#类型
t向量2 = ti.types.vector(2, float)
t向量3 = ti.types.vector(3, float)
t向量4 = ti.types.vector(4, float)
#===============================================================================
# 函数
#===============================================================================
@ti.func
def lerp(a, b, t):
	return a + (b - a) * t
@ti.func
def random_vector3():	#随机向量3
	return t向量3(ti.random(), ti.random(), ti.random())
@ti.func
def random_in_unit_sphere():
	θ = 2.0 * PI * ti.random()
	φ = ti.acos((2.0 * ti.random()) - 1.0)
	r = ti.pow(ti.random(), 1.0/3.0)
	return t向量3(r * ti.sin(φ) * ti.cos(θ), r * ti.sin(φ) * ti.sin(θ), r * ti.cos(φ))
	# p = 2.0 * random_vector3() - ti.Vector([1, 1, 1])
	# while p.norm() >= 1.0:
	# 	p = 2.0 * random_vector3() - ti.Vector([1, 1, 1])
	# return p
@ti.func
def random_unit_vector():
	return random_in_unit_sphere().normalized()
@ti.func
def reflect(v, n):	#反射
	return v - 2 * v.dot(n) * n
@ti.func
def refract(uv, n, etai_over_etat):	#折射
	cosθ = min(n.dot(-uv), 1.0)
	r_out_perp = etai_over_etat * (uv + cosθ * n)
	r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
	return r_out_perp + r_out_parallel
@ti.func
def reflectance(cosine, ref_idx):	#反射比
	# 施力克近似（Schlick's approximation）求反射透明度
	r0 = (1 - ref_idx) / (1 + ref_idx)
	r0 = r0 * r0
	return r0 + (1 - r0) * pow((1 - cosine), 5)
#===============================================================================
# 几何体
#===============================================================================
@ti.data_oriented
class C球体:
	def __init__(self, a位置: t向量3, a半径: float, a颜色: t向量3, a材质):
		self.m位置 = a位置
		self.m半径 = a半径
		self.m颜色 = a颜色	#物体反射光
		self.m材质 = a材质
	@ti.func
	def hit(self, a光线, a最小值, a最大值):	#求光线交于圆表面的距离,不相交则返回0
		v相对位置 = a光线.m位置 - self.m位置
		a = a光线.m方向.dot(a光线.m方向)
		b = 2 * v相对位置.dot(a光线.m方向)
		c = v相对位置.dot(v相对位置) - self.m半径 * self.m半径
		d = b * b - 4 * a * c
		x = 0.0
		if d > 0.0:
			sqrt_d = ti.sqrt(d)
			x = (-b - sqrt_d) / (2 * a)
			if x <= a最小值 or x >= a最大值:
				x = (-b + sqrt_d) / (2 * a)
				if x <= a最小值 or x >= a最大值:
					x = 0.0
		else:
			x = 0.0
		#返回
		v碰撞 = False
		v交点 = t向量3(0, 0, 0)
		v交点法线 = t向量3(0, 0, 0)
		v前面 = False
		if x > 0.0:
			v碰撞 = True
			v交点 = a光线.at(x)
			v交点法线 = (v交点 - self.m位置).normalized()
			if a光线.m方向.dot(v交点法线) < 0:
				v前面 = True
			else:
				v交点法线 = -v交点法线
		return v碰撞, x, v交点, v交点法线, v前面, self.m颜色, self.m材质
@ti.data_oriented
class C矩形:	#平面的
	def __init__(self, a位置: t向量3, a尺寸: t向量2, a旋转: float, a颜色: t向量3, a材质):
		self.m位置 = a位置
		self.m半尺寸 = a尺寸 * 0.5
		self.m旋转 = a旋转	#简化计算,绕y轴旋转
		self.m颜色 = a颜色
		self.m材质 = a材质
	@ti.func
	def hit(self, a光线, a最小值, a最大值):
		#旋转世界,把图片放置于y-z平面
		c = ti.cos(-self.m旋转)
		s = ti.sin(-self.m旋转)
		v旋转矩阵 = ti.Matrix([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
		v相对位置 = a光线.m位置 - self.m位置
		v相对位置 = v旋转矩阵 @ v相对位置
		v相对方向 = v旋转矩阵 @ a光线.m方向
		#计算相交
		v碰撞 = False
		t = 0.0
		v交点 = t向量3(0.0, 0.0, 0.0)
		v交点法线 = t向量3(0.0, 0.0, 0.0)
		v前面 = False
		if v相对方向.x != 0:	#光线和矩形不平行才能相交
			#有2种情况:
			#1:相对位置是负,为了相交,相对方向应该是正的
			#2:相对位置是正,为了相交,相对方向应该是负的
			t = -v相对位置.x / v相对方向.x	#如果存在相交的情况,t应该是正的
			v相对交点 = v相对位置 + v相对方向 * t
			if t <= a最小值 or t >= a最大值:
				pass
			elif abs(v相对交点.z) < self.m半尺寸.x and abs(v相对交点.y) < self.m半尺寸.y:	#交点在矩形内
				v碰撞 = True
				v交点 = a光线.at(t)
				v交点法线 = t向量3(c, 0.0, s)
				if v相对位置.x >= 0:
					v前面 = True
				else:
					v交点法线 = -v交点法线
		return v碰撞, t, v交点, v交点法线, v前面, self.m颜色, self.m材质
@ti.data_oriented
class C图片:	#平面的
	def __init__(self, a位置: t向量3, a尺寸: t向量2, a旋转: float, a路径: str):
		self.m位置 = a位置
		self.m半尺寸 = a尺寸 * 0.5
		self.m旋转 = a旋转	#简化计算,绕y轴旋转
		#纹理
		v文件 = Image.open(a路径)
		if v文件.mode != "RGBA":
			raise ValueError("只能载入带透明通道的图片")
		v数据 = np.array(v文件.getdata(), dtype = float)
		v数据 /= 255.0
		v填充 = np.zeros((v文件.width, v文件.height, 4), dtype = float)
		for x in range(v文件.width):
			for y in range(v文件.height):
				v填充[x, y] = v数据[y * v文件.width + x]
		self.m纹理 = ti.Vector.field(4, dtype = float, shape = (v文件.width, v文件.height))
		self.m纹理.from_numpy(v填充)
		v文件.close()
	@ti.func
	def hit(self, a光线, a最小值, a最大值):
		#旋转世界,把图片放置于y-z平面
		c = ti.cos(-self.m旋转)
		s = ti.sin(-self.m旋转)
		v旋转矩阵 = ti.Matrix([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
		v相对位置 = a光线.m位置 - self.m位置
		v相对位置 = v旋转矩阵 @ v相对位置
		v相对方向 = v旋转矩阵 @ a光线.m方向
		#计算相交
		v碰撞 = False
		t = 0.0
		v交点 = t向量3(0.0, 0.0, 0.0)
		v交点法线 = t向量3(0.0, 0.0, 0.0)
		v颜色 = t向量3(0.0, 0.0, 0.0)
		if v相对位置.x <= 0.0:	#光线在图片背面,不计算
			pass
		elif v相对方向.x >= 0.0:	#光线与y-z平面平行或背向y-z平面,不可能相交
			pass
		else:
			t = -v相对位置.x / v相对方向.x
			v相对交点 = v相对位置 + v相对方向 * t
			if t <= a最小值 or t >= a最大值:
				pass
			elif abs(v相对交点.z) < self.m半尺寸.x and abs(v相对交点.y) < self.m半尺寸.y:	#交点在矩形内
				#纹理采样
				v纹理尺寸x = float(self.m纹理.shape[0])
				v纹理尺寸y = float(self.m纹理.shape[1])
				v纹理坐标x = v纹理尺寸x - (v相对交点.z / self.m半尺寸.x * 0.5 + 0.5) * v纹理尺寸x - 1
				v纹理坐标y = v纹理尺寸y - (v相对交点.y / self.m半尺寸.y * 0.5 + 0.5) * v纹理尺寸y - 1
				v纹理坐标x0 = int(ti.floor(v纹理坐标x))
				v纹理坐标x1 = int(ti.ceil(v纹理坐标x))
				v纹理坐标y0 = int(ti.floor(v纹理坐标y))
				v纹理坐标y1 = int(ti.ceil(v纹理坐标y))
				v纹理坐标x_ = v纹理坐标x - v纹理坐标x0
				v颜色0 = lerp(self.m纹理[v纹理坐标x0, v纹理坐标y0], self.m纹理[v纹理坐标x1, v纹理坐标y0], v纹理坐标x_)
				v颜色1 = lerp(self.m纹理[v纹理坐标x0, v纹理坐标y1], self.m纹理[v纹理坐标x1, v纹理坐标y1], v纹理坐标x_)
				v颜色2 = lerp(v颜色0, v颜色1, v纹理坐标y - v纹理坐标y0)	#双线性纹理过滤
				#根据颜色透明度,决定碰撞还是穿过
				if ti.random() <= v颜色2.w:
					v碰撞 = True
					v交点 = a光线.at(t)
					v交点法线 = t向量3(c, 0.0, s)
					v颜色 = t向量3(v颜色2.x, v颜色2.y, v颜色2.z)
		return v碰撞, t, v交点, v交点法线, True, v颜色, E材质.e漫反射
#===============================================================================
# 渲染
#===============================================================================
class E材质(enum.IntEnum):
	e光源 = enum.auto()
	e漫反射 = enum.auto()
	e镜面反射 = enum.auto()
	e折射 = enum.auto()
	e模糊反射 = enum.auto()	#带点模糊的镜面反射
@ti.data_oriented
class C光线:
	def __init__(self, a位置: t向量3, a方向: t向量3):
		self.m位置 = a位置
		self.m方向 = a方向
	@ti.func
	def at(self, t: float):
		return self.m位置 + self.m方向 * t
@ti.data_oriented
class C场景:	#存放物体
	def __init__(self):
		self.m物体 = []
	def add(self, obj):
		self.m物体.append(obj)
	@ti.func
	def hit(self, a光线):	#计算光与哪个物体相交,返回物体属性
		v距离 = c最大值
		v碰撞 = False
		v交点 = t向量3(0, 0, 0)
		v交点法线 = t向量3(0, 0, 0)
		v前面 = False
		v颜色 = t向量3(0, 0, 0)
		v材质 = E材质.e漫反射
		for i in ti.static(range(len(self.m物体))):
			v碰撞0, v距离0, v交点0, v交点法线0, v前面0, v颜色0, v材质0 = self.m物体[i].hit(a光线, c最小值, v距离)
			if v碰撞0:
				v距离 = v距离0
				v碰撞 = v碰撞0
				v距离 = v距离0
				v交点 = v交点0
				v交点法线 = v交点法线0
				v前面 = v前面0
				v颜色 = v颜色0
				v材质 = v材质0
		#返回
		return v碰撞, v交点, v交点法线, v前面, v颜色, v材质
@ti.data_oriented
class C相机:	#生成光线
	def __init__(self, a位置: t向量3, a目标: t向量3, a上方: t向量3):
		self.m位置 = a位置
		self.m目标 = a目标
		self.m上方 = a上方
@ti.data_oriented
class C透视投影:
	def __init__(self, a宽高比: float = 1, a视角: float = PI / 3):
		self.m宽高比 = a宽高比
		self.m视角 = a视角
@ti.data_oriented
class C取景框:	#由相机和透视投影计算得到
	def __init__(self, a相机: C相机, a透视: C透视投影):
		self.m位置 = a相机.m位置
		v半高 = ti.tan(a透视.m视角 / 2)
		v半宽 = v半高 * a透视.m宽高比
		w = (a相机.m位置 - a相机.m目标).normalized()
		u = (a相机.m上方.cross(w)).normalized()
		v = w.cross(u)
		self.m左下角 = a相机.m位置 - v半宽 * u - v半高 * v - w
		self.m水平 = 2 * v半宽 * u
		self.m垂直 = 2 * v半高 * v
	@ti.func
	def get_ray(self, u: float, v: float):
		v位置 = self.m位置
		v方向 = self.m左下角 + u * self.m水平 + v * self.m垂直 - self.m位置
		v光线 = C光线(v位置, v方向)
		return v光线