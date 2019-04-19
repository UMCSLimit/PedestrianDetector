class Blob:
    xHistory = []
    yHistory = []
    def __init__(self, x, y):
        self.xHistory.append(x)
        self.yHistory.append(y)

    def newPosition(self, x, y):
        self.xHistory.append(x)
        self.yHistory.append(y)

    def sredniaX(self):
        for i in ranges
        return sum(self.xHistory) / len(self.xHistory)
    
    def sredniaY(self):
        return sum(self.yHistory) / len(self.yHistory)