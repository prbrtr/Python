import turtle 

hight=int(input("input hight/10 (odd num) "))
length=int(input("input length "))
wn = turtle.Screen()
t=turtle.Turtle()
wn.bgcolor('black')
t.color("green")
t.speed(1000)

def square(t,hs,hi):
    '''turtle,horrizental lines'''
    t.forward (hs)
    ls=hi*10
    for i in range(hi):#print horrizental lines
        if i % 2 == 0:
            t.left(90)
            t.forward(10)
            t.left(90)
            t.forward(hs*2)
        else:
            t.right(90)
            t.forward(10)
            t.right(90)
            t.forward(hs*2)
    t.left(90)
    t.forward(ls)
    for i in range(int(hs/5)):#print vertical lines
        if i % 2 == 0:
            t.left(90)
            t.forward(10)
            t.left(90)
            t.forward(ls)
        else:
            t.right(90)
            t.forward(10)
            t.right(90)
            t.forward(ls)

square(t,length,hight)
t.penup()
t.forward(hight*10)
t.pendown()
t.left(90)
t.backward (length)
square(t,length,hight)





wn.mainloop()
