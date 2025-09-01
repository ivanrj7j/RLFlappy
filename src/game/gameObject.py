class GameObject:
    def __init__(self, x:float, y:float, w:float, h:float, asset:str=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.asset = None