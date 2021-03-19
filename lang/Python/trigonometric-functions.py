
from math import degrees, radians, sin, cos, tan, asin, acos, atan, pi
rad, deg = pi/4, 45.0
print("Sine:", sin(rad), sin(radians(deg)))
print("Cosine:", cos(rad), cos(radians(deg)))
print("Tangent:", tan(rad), tan(radians(deg)))
arcsine = asin(sin(rad))
print("Arcsine:", arcsine, degrees(arcsine))
arccosine = acos(cos(rad))
print("Arccosine:", arccosine, degrees(arccosine))
arctangent = atan(tan(rad))
print("Arctangent:", arctangent, degrees(arctangent))

