import math

class Point:
    x = 0
    y = 0
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def calculateDistance(p1, p2):
        print("distance ", math.sqrt( ((p1.x - p2.x )**2) + ((p1.y - p2.y)**2 ) ))
        return math.sqrt( ((p1.x - p2.x )**2) + ((p1.y - p2.y)**2 ) )
    
class Blob:
    nearDistance = 70
    founded = True
    points = []
    vectors = []

    def __init__(self, x, y):
        self.vectors.append(Point(0, 0))
        self.points.append(Point(x, y))

    def newPosition(self, x, y):
        lastPoint = self.getLastPoint()
        self.vectors.append(Point(x - lastPoint.x, y - lastPoint.y))
        self.points.append(Point(x, y))

    def getLastPoint(self):
        return self.points[-1]

    def isNearToLast(self, p):
        print("to near ", Point.calculateDistance(p, self.getLastPoint()))
        if Point.calculateDistance(p, self.getLastPoint()) < Blob.nearDistance:
            return True
        else:
            return False

    @staticmethod
    def isNear(p1, p2):
        print("XDDD  ",p1.x)
        if Point.calculateDistance(p1, p2) < Blob.nearDistance:
            return True
        else:
            return False

