class Birds:
    def __init__(self, name):
        self.name = name 
    def hello(self):
        print("Я птица",self.name)

b1 = Birds("Руслан")
b2 = Birds("Людмила")

b1.hello()
b2.hello()