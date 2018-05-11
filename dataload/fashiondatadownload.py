import os
from shutil import which #To check if SVN exists in path

####################################################################
# This class uses SVN as a hack to download data from a specific folder on github. 
# SVN is used, as GIT itself does not itself support checkout of a subfolder (at least in May 2018)
# /Aries
####################################################################
class FashionDownload:

    def downloadIfNotExist():
        if not os.path.isdir('data/fashion'):
            if which('svn') is not None:
                print("SVN is installed! Good..")
		        #Just create data folder if it does not exist, but it should. :-)
                if not os.path.exists('data'):
                    os.makedirs('data')
                if not os.path.isdir('data/fashion'):
                    print('Downloading Fashion MNIST with SVN workaround from github')
			        #Will also create fashion folder
                    os.system('svn export https://github.com/zalandoresearch/fashion-mnist/trunk/data/fashion/ data/fashion')
            else:
                print("SVN not installed")
                print("You will to manually download the fashion MNIST files from https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion") 
                print("and add them to the /data/fashion/ folder")
        else:
             print('/data/fashion folder exists, skipping download of fashion MNIST data')






