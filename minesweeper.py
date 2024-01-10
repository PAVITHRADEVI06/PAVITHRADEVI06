from tkinter import*#imports a library
import settingsmin #imports a file
import utilitiesmin
from cellmin import cell

root = Tk() #tk() is a window and root a variable that acts as pointer and tk is an object which contains data
root.configure(bg="black")
root.geometry(f'{settingsmin.WIDTH}x{settingsmin.HEIGHT}')
root.title("Minesweeper Game")
root.resizable(False, False) # one false for width and another for height
top_frame = Frame(
    root,#where this frame needs to be used or positioned
    bg='black',#later change it to black
    width=settingsmin.WIDTH,
    height=utilitiesmin.height_prct(25)
)
top_frame.place(x=0,y=0)#place describes the place from which the frame starts from in terms of pixels

left_frame = Frame(
    root,
    bg ='black',#change later to black
    width=utilitiesmin.width_prct(25),
    height=utilitiesmin.height_prct(75)
)
left_frame.place(x=0,y=180)

center_frame = Frame(
    root,
    bg='black',
    width=utilitiesmin.width_prct(75),
    height=utilitiesmin.height_prct(75)
)
center_frame.place(
    x=utilitiesmin.width_prct(25),
    y=utilitiesmin.height_prct(25),
)

for x in range(settingsmin.GRID_SIZE):
    for y in range(settingsmin.GRID_SIZE):
        c=cell(x,y)
        c.create_btn_object(center_frame)
        c.cell_btn_object.grid(
            column=x,row=y
)

print(cell.all)
        
#c1=cell()
#c1.create_btn_object(center_frame)
#c1.cell_btn_object.grid(
#    column=0,row=0
#)


#assigning events to buttons=a list of actions to take once a button is clicked
#for example different actions to be taken when you left or right click on a button





root.mainloop()#program is written within this, tells till when it should run
