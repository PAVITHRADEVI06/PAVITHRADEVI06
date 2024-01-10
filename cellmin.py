from tkinter import Button
class cell:
    all=[]
    def __init__(self,x,y,is_mine=False): #constructor is going to be called immedietly after a class is initiated
        self.is_mine=is_mine #self is to use the instances in a class and __init__ is used to initiate constructor
        self.cell_btn_object=None
        self.x=x
        self.y=y

    #append the object to the cell.all list
        cell.all.append(self)
    def create_btn_object(self, location): 
        btn=Button(
            location,
            width=12,
            height=4,
            text=f"{self.x},{self.y}"

        )
        btn.bind('<Button-1>',self.left_click_actions) #with bind we say we'd like to print something when we left/right click on a button
        btn.bind('<Button-3>',self.right_click_actions)
        self.cell_btn_object=btn

    def left_click_actions(self,event):
        print(event)
        print("i am left clicked!")

    def right_click_actions(self,event):
        print(event)
        print("i am right clicked!")

    @staticmethod
    def randomize_mines():
        pass


    def __repr__(self): #changes the way object is being represented
        return f"cell({self.x},{self.y})"
        
#instances are individual object of a particular class
#instantiation creating an instance of a class
#instance method a special kind of function that is defined in a class definition
#is_mine is used to determine whether a given button is a mine or not
#anything inside def() is class definition and self.instance is assigning a value to the variables declared in a function definition
#static to commonly use instances globally in class at any functions in a class
