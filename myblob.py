import math

class Blob:
    nearDistance = 60;
    founded = True
    xHistory = []
    yHistory = []

    def __init__(self, x, y):
        self.xHistory.append(x)
        self.yHistory.append(y)

    def newPosition(self, x, y):
        self.xHistory.append(x)
        self.yHistory.append(y)

    def sredniaX(self):
        return sum(self.xHistory) / len(self.xHistory)
    
    def sredniaY(self):
        return sum(self.yHistory) / len(self.yHistory)

    def distance(self, x1, y1, x2, y2):
        #print("deltaX = ", x1 - x2)
        #print("deltaY = ", y1 - y2)
        dis = math.sqrt( ((x1 - x2)**2) + ((y1 - y2)**2) )
        #print("Distance = ", dis)
        return dis

    def isNear(self, xK, yK):
        distancePoints = self.distance(self.xHistory[-1], self.yHistory[-1], xK, yK)
        if distancePoints < self.nearDistance:
            return True
        else:
            return False
