import supervision as sv
import inspect

print(f"Supervision Version: {sv.__version__}")
print("LineZoneAnnotator.__init__ signature:")
try:
    print(inspect.signature(sv.LineZoneAnnotator.__init__))
except Exception as e:
    print(e)
