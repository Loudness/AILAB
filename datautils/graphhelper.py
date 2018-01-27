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
    
    
    #plt.figure(figsize=(30,50))
    plt.tight_layout(pad=15)
    # plot original series
    plt.plot(original_data_set,color = 'k')
    

    # plot training set prediction
    split_pt = train_test_split + window_size 
    plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')

    # plot testing set prediction
    plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')

    # pretty up graph
    plt.xlabel('Days')
    plt.ylabel('(normalized) price of SPY')
    #plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(['original series','training fit','testing fit'],loc='lower center', bbox_to_anchor=(1, 0.5))
    	
   
    #plt.savefig(imagename, bbox_inches='tight',  dpi = (400))         #bbox_inches is a hack to get the legend fit in the image
    plt.savefig(imagename)         #bbox_inches is a hack to get the legend fit in the image
    #plt.show()
    #fig = plt.figure()
    #plt.close(fig) 
    #fig.savefig(imagename)
   
    
