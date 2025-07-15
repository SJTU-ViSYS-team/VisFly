from VisFly.utils.maths import Quaternion


a = Quaternion.from_euler(30/57.3, 0,0)
print(a.R)
print(a.xz_axis)