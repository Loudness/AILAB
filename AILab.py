


# This is a playground for testing different AI techniques. 
# by Aries 
# It will focus on explanation and progress of the techniques
# Motivation:
# Why have an AI Lab? 
# Most of the AI techniques are centrered on reading data, validating data, processing data and then trying to make sense of the result.
# This means we are repeating a lot of the stages, If we could re-use some of the data reading, or validation at the end it would speed up our development cycle. 
# At least at the testing stage. Then export the code to any system.
# It could have been done with Jupyter Notebooks, but the idea is to do it in a native environment. 
# Python 3.6
# Keras
# Scikit-learn
# ..and so on
# Install dependencies with 'pip install -r requirements.txt'

import os
import sys
import msvcrt as m
from datautils import sysinfo
import LSTM_Lab1
import GAN_Lab1
import MachineTransation1
import SentimentAnalysis1

if (os.name=='nt'):
    import msvcrt as myGetch
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
else:
    import getch as myGetch

# Function for waiting for key press
def wait():myGetch.getch()


# Clear screen for windoze and linux/mac
def cls():os.system('cls' if os.name=='nt' else 'clear')
 
cls()

ans=True
showAtStartOnly = True
while ans:
    if showAtStartOnly:
        print("""
        ###################################################################################
        # Welcome to AI Lab 
        # This is a playground for testing AI techniques.
        ###################################################################################
        # It is kept simple and uses components for reading and prerocessing data, 
        # validation, running algos, and finally showing results.
        # The idea is to research ideas and explaing roughly what does what. 
        # This way someone else can just re-run it and see the flow of using it and learning on the way.
        # There are no rules how to use it except that it should be reproducable by anyone with a few commands. 
 
        ###################################################################################
        """)
        showAtStartOnly = False

    
    print("""
    Menu:
    ------------
    1.Run LSTM Lab  - Time series Prediction based on 2500 days on SPY: SPDR S&P 500
    2.Run GAN Lab   - Generate images from Fashion Images.
    3.Run Statistical Machine Translation Lab  - Translate from English to French
    4.Run Sentiment Analysis Lab - Write a positive or negative movie review - Base on IMDB data
    5.Next lab here
    9.Show System Info
    0.Exit/Quit
    """)
    ans=input("What would you like to do? ")
    if ans=="1":
      LSTM_Lab1.run_lstmLab1()
      print("\nFinished LSTM LAB 1, check newly created folder for saved results")
      print("\nPress Enter...")
      wait()
      cls()

    elif ans=="2":
      GAN_Lab1.run_GANLab1()
      print("\nPress Enter...")
      wait()
      cls()
    elif ans=="3":
      MachineTransation1.run_machineTranslationLab1()
      
      print("\nPress Enter...")
      wait()
      cls()
    elif ans=="4":
      SentimentAnalysis1.run_sentimentLab1()
      print("\nFinished Sentiment analysis lab.")
      print("\nPress Enter...")
      wait()
      cls()
    elif ans=="5":
      print ("Test seaborn output")
      LSTM_Lab1.run_example()
      print("\nNext lab here")
      print("\nPress Enter...")
      wait()
      cls()
    elif ans=="6":
      LSTM_Lab1.createGraph()
      LSTM_Lab1.testResultToBrowser()
      print("\nPress Enter...")
      wait()
      cls()
    elif ans=="9":
      print ("Fetching system info..")
      sysinfo.showSystemInfo()
      print("\nPress Enter...")
      wait()
      cls()

    elif ans=="0":
      print("\nGoodbye") 
      ans = None
    else:
      print("\nNot Valid Choice Try again")
      print("\nPress Enter...")
      wait()
      cls()
      ans = True



