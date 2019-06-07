# Malaria Detection Using Computer Vision
## This Repository Is Under Construction
Artificial intelligence (AI) has the potential to help tackle some of the world’s most challenging world problems and when 
coupled with popular tools and technologies  for development and betterment of our society ,what the point of techology when it can't help the needy and save lives. Deep learning helps us to  build robust, scalable and effective solutions which can be adopted by everyone even in remote corners of the world and detection of Malaria is one of the problem which Deep learning has help to to tackle.

Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. It is preventable and curable.According to WHO their are 212 Million malaria cases and 435000 deaths.Early diagnostics and treatment of malaria can prevents deaths.Malaria is prevalent across the world especially in tropical regions
The severity of malaria varies based on the species of plasmodium.
Symptoms are chills, fever and sweating, usually occurring a few weeks after being bitten.

If efficent detection of malaria is made then deaths can be avoided which will ave families and communities from downward spiral of poverty.
Most of the 435000 who died of Malaria were children, mainly in Africa, which is hyperendemic for malaria.when severe malaria does occur, malnourished children have a higher morbidity and mortality.when severe malaria does occur, malnourished children have a higher morbidity and mortality.when severe malaria does occur, malnourished children have a higher morbidity and mortality.
Malaria  causes Renal failure,Pulmonary oedema,Jaundice,Anemia,Pre-treatment hypoglycemia and Neurological sequelae and Convulsions


![poor-people-fo-real](https://user-images.githubusercontent.com/37455387/58870226-6ca29700-86dd-11e9-9485-9bdd0b5f37ac.jpg)

### Impact Of Malaria On The World
![1_HHQlfj2REThOojQLNhnUZw](https://user-images.githubusercontent.com/37455387/58866717-c6ec2980-86d6-11e9-811b-d4e0a9c922c9.png)

Malaria is one of the world’s deadliest diseases, and remains one of the top child killers on the planet. Malaria also keeps children from going to school, families from investing in their future, and communities from prospering, taking a huge toll on lives, livelihoods and countries’ progress.

## Microscopic Diagnosis

Malaria parasites can be identified by examining under the microscope a drop of the patient’s blood, spread out as a “blood smear” on a microscope slide. Prior to examination, the specimen is stained to give the parasites a distinctive appearance. This technique remains the gold standard for laboratory confirmation of malaria. However, it depends on the quality of the reagents, of the microscope, and on the experience of the laboratorian.WHO recommends that all cases of suspected malaria be confirmed using parasite-based diagnostic testing (either microscopy or rapid diagnostic test) before administering treatment. Results of parasitological confirmation can be available in 30 minutes or less.

![7799cb57fdd2d20e1b7509dcec6dff19_whatismalaria_malaria_parasite_585-970-1100-c-90](https://user-images.githubusercontent.com/37455387/58872006-bb9dfb80-86e0-11e9-8a9c-8cf558151ee7.jpg)

 
The research  paper on which the data and analysis is constructed , ‘ Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images’ by S Rajaraman et. al. introduces to some of these methods. These include thick and thin blood smear examinations, polymerase chain reaction (PCR) and rapid diagnostic tests (RDT) the latter two tests are surrogate methods  used an  alternative particularly where good quality microscopy services cannot be readily provided.

## Convolution Neural Network

![The-convolutional-neural-network-CNN-architecture-for-the-deep-learning-based-cartilage](https://user-images.githubusercontent.com/37455387/58866893-1cc0d180-86d7-11e9-9cba-3fff3ae3be6d.png)

Convolution neural networks are special type of neural networks used in images recognition, images classifications and  Objects detections.A convolutional neural network consists of an input and an output layer, as well as multiple hidden layers. The hidden layers of a CNN typically consist of convolutional layers, RELU layer i.e. activation function, pooling layers, fully connected layers and normalization layers.

Convolutional layer is core building block of CNN, it helps with feature detection.

Kernel K is a set of learnable filters and is small spatially compared to the image but extends through the full depth of the input image.

An easy way to understand this is if you were a detective and you are came across a large image or a picture in dark, how will you identify the image?

You will use you flashlight and scan across the entire image. This is exactly what we do in convolutional layer.

Kernel K, which is a feature detector is equivalent of the flashlight on image I, and we are trying to detect feature and create multiple feature maps to help us identify or classify the image.

we have multiple feature detector to help with things like edge detection, identifying different shapes, bends or different colors etc

![1_XbuW8WuRrAY5pC4t-9DZAQ](https://user-images.githubusercontent.com/37455387/58934834-3fabbe00-8789-11e9-81bf-5950375c5757.jpeg)



## Tools
![1_Q9L4auKM6DbpV2BH32GnoQ](https://user-images.githubusercontent.com/37455387/58870251-7926ef80-86dd-11e9-9980-8e47fb3be22a.jpeg)

The fast Ai library and pytorch is used, the architectures used are ResNet34,ResNet50 and ResNet152 and the models were trained on google colab which uses Tesla K80 GPU and total 12 Gigabytes of RAM. 

![s_8C760A111A4204FB24FFC30E04E069BD755C4EEFD62ACBA4B54BBA2A78E13E8C_1490999744884_main-qimg-b1fcbef975924b2ec4ad3a851e9f3934](https://user-images.githubusercontent.com/37455387/58935314-cad98380-878a-11e9-99e6-934d0bda2cec.png)


A network which produces x amount of training error. Construct a network B by adding few layers on top of A and put parameter values in those layers in such a way that they do nothing to the outputs from A. Let’s call the additional layer as C. This would mean the same x amount of training error for the new network. So while training network B, the training error should not be above the training error of A. And since it DOES happen, the only reason is that learning the identity mapping(doing nothing to inputs and just copying as it is) with the added layers-C is not a trivial problem, which the solver does not achieve. To solve this, the module shown above creates a direct path between the input and output to the module implying an identity mapping and the added layer-C just need to learn the features on top of already available input. Since C is learning only the residual, the whole module is called residual module.
ResNet34 has network depth of 34, ResNet50 has network depth of 50 and ResNet152 has network depth of 152 layers.

### Freezing and Unfreezing Layers
When doing transfer learning we typically freeze the first n layers and leave the last layer unfrozen to be able to update the weights.
If the first N layers are frozen that means that if we put an image through it during the first epoch and we put the same image through it again during the second epoch then we will get out the same value through that layer.

Consider a network that has 2 layers. The first layer is frozen and the second layer not frozen. If we run 100 epochs we are doing an identical computation through the first layer for each of the 100 epochs. We run the same images through the same layers without updating the weights. this means for every epoch the inputs to the first layer are the same(the images). The weights in the first layer are the same and the outputs from the first layer are the same(images * weights + bias).

So instead of running that same calculation for each epoch we only run it once. then we feed that output into layer 2. That output is also called the activation.

### Optimal Learning Rate

The learning rate is the most important hyper-parameter for training neural networks, yet until recently deciding its value has been incredibly shady.. 
We do a trial run and train the neural network using a low learning rate, but increase it exponentially with each batch
Meanwhile, the loss is recorded for every value of the learning rate. We then plot loss against learning rate: like below
.
![download](https://user-images.githubusercontent.com/37455387/59104682-c8745680-894f-11e9-8251-cc78c0566b97.png)

The optimum learning rate is determined by finding the value where the learning rate is highest and the loss is still descending


REFERNCES:
[1] https://www.who.int/news-room/fact-sheets/detail/malaria

[2] https://www.cdc.gov/malaria/diagnosis_treatment/diagnosis.html

[3]https://towardsdatascience.com/detecting-malaria-with-deep-learning-9e45c1e34b60 

[4]https://medium.com/datadriveninvestor/convolutional-neural-network-cnn-simplified-ecafd4ee52c5

[5]https://github.com/krishnakatyal/Malaria-Detection-with-Deep-Learning/blob/master/Deep_learning_for_Malaria_detection.ipynb
