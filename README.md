# SplitEasy

This repository contains the code for our paper [SplitEasy: A Practical Approach for Training ML models on Mobile Devices in a split second](https://arxiv.org/abs/2011.04232). 

The etnire code is written using React Native. To follow the general guideline for React Native apps follow this [link](https://developers.facebook.com/docs/react-native/getting-started/). 

There are three parts to this code: loading the dataset on the mobile device, running the models on the mobile device and running the models on the server. 


### Loading the dataset on the mobile device

The data files will be available in this [link](). Move them to your assets folder. The json files are split into 5 parts each with 25 images, so that we can load the image into the phone. 

Once the data is in the assets folder, go to *App.js* and ensure that you load the *LoadImageNet* component. The code will load take some time to load on your device, once it loaded click the load data button. The data that is being loaded will be logged in your terminal. Then, change the i value in line 58 of *imagenet_load.js* to i
+25 and change the name of the file to the next file. Repeat this process for 5 times and the data is loaded in your phone.

### Loading the dataset on the mobile device

The component to load in your *App.js* now is the *SplitNet* component. The main changes to make in this component is to add the server url in the *App.js* file and *final_implementation.js* file. Based on the model, load the appropriate file for ModelA and ModelC, the files are availabe in this [link](). The component will have the button for training the model and once you click it the values are logged in your terminal. 

### Running the Server Code

The server code should preferably be run in a server with GPU support. The code requires the argument *--model_name* in which you mention the name of the architecture. Once you run the code, you can use the IP address of your server to run the javascript code. 

