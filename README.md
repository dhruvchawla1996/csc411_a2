# CSC411
Machine Learning and Data Mining - University of Toronto Winter 2018

Use Python2

[cropped227.zip](https://drive.google.com/open?id=1ctcwF9IRBD68ycKUE52yyXiio9S44X51)

[cropped64.zip](https://drive.google.com/open?id=1a23dvrrKjfDTZ5IMWs8w9_vOch7er_yq)

## Project 2
Deep Neural Networks for Handwritten Digit and Face Recognition

### Part 1
Load ```digits.py``` and call ```part1()```

Stores image in ```figures/part1.png```

Needs ```mnist_all.mat``` and ```snapshot50.pkl``` to be present

### Part 2
Load ```digits.py``` and call ```part2()```

Returns a 10x1 numpy array containing the output of the NN

Needs ```mnist_all.mat``` and ```snapshot50.pkl``` to be present

### Part 3
Load ```digits.py``` and call ```part3()```

Outputs values of gradient along the ```coords``` vector along with the finite difference calculation along the same ```coords```

Needs ```mnist_all.mat``` and ```snapshot50.pkl``` to be present

### Part 4
Load ```digits.py``` and call ```part4()```

Prints out the training and testing performance along with epochs 

Saves learning curve graph in ```figures/part4_learning_curve.png``` and the digit weights in ```figures/part4_i.png``` where ```i``` is 0 to 9

Needs ```mnist_all.mat``` and ```snapshot50.pkl``` to be present

### Part 5
Load ```digits.py``` and call ```part5()```

Prints out the training and testing performance along with epochs 

Saves learning curve graph in ```figures/part5_learning_curve.png```

Needs ```mnist_all.mat``` and ```snapshot50.pkl``` to be present

### Part 6


### Part 7
No code, everything is in ```deepnn.pdf```

### Part 8
Load ```deepfaces.py``` and call ```part8()```

Prints out the training and testing performance along with epochs 

Saves learning curve graph in ```figures/part8_learning_curve.png```

Saves weights and bias going into the hidden layer in ```part8_model_params.pkl```

Needs 64x64 RGB images of each of the actors in ```cropped64``` folder

If the above is not present, uncomment the code in ```part8()``` to get the images or unzip ```cropped64.zip```

If you decide to run the code, you will need ```faces_subset.txt``` file

### Part 9
Load ```deepfaces.py``` and call ```part9()```

Saves figures in ```figures/bracco``` and ```figures/baldwin```

Needs ```part8_model_params.pkl```

Needs ```cropped64/bracco27.jpg``` and ```cropped64/baldwin38.jpg``` (should be there from Part 8)

### Part 10
Load ```deepfaces.py``` and call ```part8()```

Prints out the training and testing performance along with epochs 

Needs 227x227 RGB images of each of the actors in ```cropped64``` folder

If the above is not present, uncomment the code in ```part10()``` to get the images or unzip ```cropped227.zip```

If you decide to run the code, you will need ```faces_subset.txt``` file
