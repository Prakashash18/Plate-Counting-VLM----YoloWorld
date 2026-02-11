import supervision as sv
import pprint

print(f"Supervision Version: {sv.__version__}")
lz = sv.LineZone(start=sv.Point(0,0), end=sv.Point(100,100))
print("Dir(lz):")
pprint.pprint(dir(lz))
print("-" * 20)
try:
    print(f"Start: {lz.start}")
except AttributeError as e:
    print(f"Error accessing .start: {e}")

try:
    print(f"Vector: {lz.vector}")
except AttributeError as e:
    print(f"Error accessing .vector: {e}")
