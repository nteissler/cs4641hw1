##Supervised Learning: Algorithm Implementations

The source code for this project can be found in the `src` folder alongside this readme. Inside is a requirements.txt. Python 3.4 or greater is required to run the code in the files and to use the dependencies. Install the dependencies from within the `src` folder with:

```
pip install -r requirements.txt
```

The datasets must also be downloaded, the Wine dataset is small enough such that it can be downloaded on the fly, each time with the python scripts. The Digits data set on the other hand must be downloaded from [here](http://yann.lecun.com/exdb/mnist/). Unzip with 

```
gzip *ubyte.gz -d
```

and save them to the **mnist** directory without changing the names. See **mnist/readme.md** for the names of the files.

Every **.py** file within `src` can be run directly with 

```
python3 <name.py>
```
with the exception of **plotter.py**, **neuralnet.py**,  **mnist_data.py** and **wine_data.py** which are modules imported for drawing the graphs with matplotlib and loading data. 

The variables within each file are documented at [scikit learn's website](http://scikit-learn.org/stable/index.html) and may be changed at will. Just make sure the code runs again. 

Upon completeion, each file should give an accuracy measure on the train and test data it classified and possibly some additional logging. 
