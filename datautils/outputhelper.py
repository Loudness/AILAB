
import os
#outputhelper to save results in a easy manner
#TODO: add Result to DB (mongoDB) method
#TODO: add plain text method
#TODO: Maybe even send a mail or SMS as an output. :-)
#TODO: Add image array as input so we can have comparative images.
#TODO: Reformat to a nicer output

def resultToBrowser(folder, image, text):
     import webbrowser
     f = open(folder + os.sep + 'result.html','w')
     message = ("""<html> <head>Results of run for """ + folder + """ </head> <body><p>""" + text + """</p> <img src=\"result.png\" alt=\"Plot graph\" width=\"512\" height=\"512\" ></body> </html>""")
     f.write(message)
     f.close()
     webbrowser.open_new_tab(folder + os.sep + 'result.html')

