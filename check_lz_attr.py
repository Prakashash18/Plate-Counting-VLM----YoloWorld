import supervision as sv
try:
    lz = sv.LineZone(start=sv.Point(0,0), end=sv.Point(100,100))
    print("Has start:", hasattr(lz, "start"))
    print("Has vector:", hasattr(lz, "vector"))
    if hasattr(lz, "vector"):
        print("Vector start:", lz.vector.start)
except Exception as e:
    print(e)
