import numpy as np
import xlrd
import xlwt 
import array as ar
from xlwt import Workbook 
loc = ("C:/Users/Specter/Desktop/test.xls")

def hardLim(x):
        if(x<=0):
                return 0
        else:
                return 1

class Neuron:
  def __init__(self, weights):
    self.weights = weights

  def feedforward(self, inputs):
    total = np.dot(self.weights, inputs)
    return hardLim(total)
  

w1 = 1
w2 = -.8
weights = np.array([w1, w2])  
n = Neuron(weights)

wb = xlrd.open_workbook(loc)
sheetr = wb.sheet_by_index(0)
sheetr.cell_value(1, 0)

count=0
wbt=Workbook()
sheet1=wbt.add_sheet('Weights')
c=0
max=sheetr.nrows

while(count!=max):
    
    for i in range(sheetr.nrows):
        print(weights)
        i1=sheetr.cell_value(i, c)
        i2=sheetr.cell_value(i,c+1)
        d=sheetr.cell_value(i,c+2)
        x = np.array([i1, i2])
        act=n.feedforward(x)
        if(act<d):
            w1=w1+i1
            w2=w2+i2
            weights=np.array([w1,w2])
            n = Neuron(weights)
            sheet1.write(i,c,w1)
            sheet1.write(i,c+1,w2)
            wbt.save('C:/Users/Specter/Desktop/out.xls')
        elif(act>d):
            w1=w1-i1
            w2=w2-i2
            weights=np.array([w1,w2])
            n = Neuron(weights)
            sheet1.write(i,c,w1)
            sheet1.write(i,c+1,w2)
            wbt.save('C:/Users/Specter/Desktop/out.xls')
        else:
            count=count+1
        print(act)
        
        
    print(weights)





import matplotlib.pyplot as plt

# x axis values
y1 = [sheetr.cell_value(0, c),sheetr.cell_value(1, c),sheetr.cell_value(2, c)]
x1=[0,1,2]
# corresponding y axis values
x2 = [0,1,2]
y2 = [sheetr.cell_value(0, c+1),sheetr.cell_value(1, c+1),sheetr.cell_value(2, c+1)]
# plotting the line 2 points

# plotting the points
plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 3,
		marker='o', markerfacecolor='blue', markersize=12)

plt.plot(x2, y2, color='red', linestyle='dashed', linewidth = 3,
		marker='o', markerfacecolor='blue', markersize=12)

# setting x and y axis range
plt.ylim(-1,3)
plt.xlim(-1,3)

# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('Some cool customizations!')

# function to show the plot
plt.show()



    

    
