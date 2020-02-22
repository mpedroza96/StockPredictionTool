from tkinter import *
from tkinter.ttk import *
import matplotlib
import Tool as Tol
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

#Style

#Window Configuration
ventana=Tk()
ventana.title("Stock Prediction Tool")
ventana.geometry("800x300")
ventana.configure(background="SkyBlue4")
ventana.grid()

#Labels
label1 = Label(ventana, text="Name a stock (e.g. 'AAPL')")
label1.configure(background="SkyBlue4", font = ('helvetica', 12))
label1.grid(row=3, column=2)

labele = Label(ventana, text=" ")
labele.configure(background="SkyBlue4")
labele.grid(row=2, column=0)
labele = Label(ventana, text=" ")
labele.configure(background="SkyBlue4")
labele.grid(row=5, column=0)

label2 = Label(ventana, text="Stock Prediction Tool")
label2.configure(background="SkyBlue4", font = ('helvetica', 20, 'bold'))
label2.grid(row=0, column=2)

#Textbox inputs
entry = Entry(ventana, text="Name a stock")
entry.grid(row=4, column=2)

#Tool class methods callouts
def stockPlot():
    stock = entry.get()
    Tol.Tool.stockPlot(Tol.Tool.stockQuote(stock))

def stockTable():
    stock = entry.get()
    Tol.Tool.stockTable(Tol.Tool.stockQuote(stock))

def trainModel():
    stock = entry.get()
    labelE = Label(ventana, text=("Model Error (RMSE): ", round(Tol.Tool.trainData(Tol.Tool.stockQuote(stock)), 2)))
    labelE.configure(background="SkyBlue4", font=('helvetica', 10))
    labelE.grid(row=8, column=2)

def predictedPlot():
    stock = entry.get()
    Tol.Tool.predictionPlot()

def predictedTable():
    stock = entry.get()
    Tol.Tool.predictionTable()

def prediction():
    labelE = Label(ventana, text=("Predicted Stock Price", list((Tol.Tool.futurePrediction())[0])))
    labelE.configure(background="SkyBlue4", font=('helvetica', 10))
    labelE.grid(row=11, column=2)


#Buttons
button1 = Button(ventana, text="Stock closing price table", command=stockTable)
button1.grid(row=6, column=1)

button2 = Button(ventana, text="Stock closing price plot", command=stockPlot)
button2.grid(row=6, column=3)

button3 = Button(ventana, text="Train ML LTSM Model (Wait 5 minutes)", command=trainModel)
button3.grid(row=7, column=2)

button4 = Button(ventana, text="Prediction closing price table (Historical Data)", command = predictedTable)
button4.grid(row=9, column=1)

button5 = Button(ventana, text="Prediction closing price plot (Historical Data)", command=predictedPlot)
button5.grid(row=9, column=3)

button6 = Button(ventana, text="Predicted closing price for tomorrow", command=prediction)
button6.grid(row=10, column=2)

#Run interface
ventana.mainloop()

