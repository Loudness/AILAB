import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#TODO: use Seaborn as in example below
#Graphhelper


def createGraphTEST(folder, inputdata=0):
    sns.set(style="whitegrid")
    networks = sns.load_dataset("brain_networks", index_col=0, header=[0, 1, 2])
    networks = networks.T.groupby(level="network").mean().T
    order = networks.std().sort_values().index
    sns.lvplot(data=networks, order=order, scale="linear", palette="mako").figure.savefig(folder + os.sep + "result.png")

#The method should save an image. This will be used afterwards in the HTML output. 
def createGraphLSTM_Lab1(imagename, window_size, original_data_set, train_test_split,train_predict, test_predict):
    
    #enable override of seaborn over pyplot/matplotlib
    sns.set()
    
    #plt.figure(figsize=(4,2))
    plt.tight_layout(pad=15)
    # plot original series
    plt.plot(original_data_set,color = 'k', linewidth=1)
    
    # plot training set prediction
    split_pt = train_test_split + window_size 
    plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b', linewidth=1)

    # plot testing set prediction
    plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r', linewidth=1)

    # pretty up graph
    plt.xlabel('Days')
    plt.ylabel('(normalized) price of SPY')
    #plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(['original series','training fit','testing fit'],loc='lower center', bbox_to_anchor=(1, 0.5))
    	         
    
    #fig = plt.gcf()
    #DefaultSize = fig.get_size_inches()
    #plt.set_figsize_inches( (DefaultSize[0]*2, DefaultSize[1]) )
    #fig.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]), forward=False )
    #plt.savefig(imagename, bbox_inches='tight',  dpi = (400))         #bbox_inches is a hack to get the legend fit in the image
    plt.savefig(imagename)          
    #plt.show()
    #fig = plt.figure()
    #plt.close(fig) 
    #fig.savefig(imagename)

""" Peek into MNIST dataset     (saved)
def showMNIST_Images(MNISTData):
    
    #import MLDatasets: MNIST
    from IterTools import product

    x, y = MNISTData.testdata()
    nrow, ncol = 4, 4
    fig = figure("plot_mnist",figsize=(6,6))
    for (i, (c, r)) in enumerate(product(1:ncol, 1:nrow))
       subplot(nrow, ncol, i)
       plt.imshow(x[:,:,i]', cmap="gray") #notice the transpose
       ax = gca()
       ax[:xaxis][:set_visible](false)
       ax[:yaxis][:set_visible](false)
    end
    tight_layout(w_pad=-1, h_pad=-1, pad=-0.5)
    plt.savefig("TEST.PNG")
"""   
    
