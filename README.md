# pyRad
Pytorch framework of inverse IMRT optimization for radiotherapy.

We hope this project served as a bridge between the deep learning and the radiotherapy community.

At this moment, we have implemented: Fluence map optimization, Direct aperture optimization, DVH objectives, Neural dose, the interface of a third-party Monte Carlo engine (gDPM), the interface of a third-party pencil beam dose engine, 3D dose representation, RTStruct parsing, RTPlan parsing. 

We did not implement our own dose engine, therefore a dose deposition file is necessary to run the code. Additionally, a dose constraint file, a Jaw position file, and a set of DICOM files are also required. We will provide these files shortly. 

The neural dose network could be found at our repository: https://github.com/lc82111/neuralDose.

Tensorbord is useful in visualizing the optimization process. Below is some screenshots.


![](imgs/1.png)
![](imgs/2.png)
![](imgs/3.png)
![](imgs/4.png)
![](imgs/5.png)
![](imgs/6.png)
![](imgs/7.png)
![](imgs/loss_total_loss.svg)


