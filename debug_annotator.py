import supervision as sv
import inspect

print(f"Supervision Version: {sv.__version__}")
print("LineZoneAnnotator.annotate signature:")
try:
    print(inspect.signature(sv.LineZoneAnnotator.annotate))
except Exception as e:
    print(e)
