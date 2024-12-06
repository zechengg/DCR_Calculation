# My Project
## Calculate the DCR parameters of SPAD using python with Comsol simulated data.
## 1. export Ionization coefficient, electrons / holes  
###   (1) create streamlines under "Electric field form" 2D-plot  
####    a. add streamline
![img_1.png](Image/img_1.png)  
####    b. add filter under streamline(adjust the figure to get the fitting streamlines)  
####    c. add Export Expressions(choose alpha_n and anlha_p)
![img_3.png](Image/img_3.png)
 ###  (2) export alpha_n & alpha_p
![img_4.png](Image/img_4.png)
## 2. export SRH generation term & BTBT generation term  
###   (1) get the points coordinate of the streamlines from step1  
      Run Ge_points_coordinate.py, then you will get the points_coordinate.txt.  
     
###    (2) export SRH & BTBT    
![img_5.png](Image/img_5.png)
## 3. Calculate the DCR parameters    
Run the DCR_calculate