import math                                                                              #math library used for math.exp() function
import csv                                                                               #csv library used to read csv file
r=open('C:\\Users\\obero\\Desktop\\Normaliseddata.csv','r')                              #reading csv file and converting into list of lists(matrix)
reader=csv.reader(r)
i=[]
o=[]
for row in reader:
    i.append([float(row[0]),float(row[1])])                                              #extracting first two rows of collected data i.e inputs
    o.append([float(row[2]),float(row[3])])                                             #extracting last two rows of collected data i.e outputs
            


train_set_i=list(i[:6100])                                                                #train,validation,test set inputs
validation_set_i=list(i[6100:7400])
test_set_i=list(i[7400:])                                                                 #ratio in which data is partitioned trainset:validation:test=70:15:15

train_set_o=list(o[:6100])                                                                #train,validation,test set targetted outputs
validation_set_o=list(o[6100:7400])
test_set_o=list(o[7400:])





class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 2
        self.hiddenSize = 6
        
        #weights
        self.w1 = [[2.7,3.3,-2.0,2.1,1.1,-2.1],[-1.9,0.3,2.6,2.1,-3.0,2.5]]            #  weight matrix from input to hidden layer (2*6 matrix)(randomly assigned)
        self.w2 =  [[0.2,-0.2],[-0.1,0.4],[0.4,0.6],[0.1,0.2],[-0.4,-0.1],[0.1,0.4]]   #  weight matrix from hidden to output layer (6*2 matrix)
        

        
        self.k_p=[[0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0]]                   #k_p and l_p these variables are instialised and used later to store previous delta-weights for momentum hyperparameter
        self.l_p=[[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]




    def sigmoid(self, s, deriv=False):                                      #activation function used is sigmoid for both input to hidden and hidden to output layers
        
        main=[]
        lis=[]
        b=False
        if (deriv == True):
                for element in s:
                        if b==True:
                                main.append(lis)
                                lis=[]
                        for index in range(len(element)):
                                lis.append(0.9*(element[index] * (1 - element[index])))
                                b=True
                if index==(len(element)-1):
                        main.append(lis)
                
            
                
        else:
            for element in s:
                    if b==True:
                            main.append(lis)
                            lis=[]
                    for index in range(len(element)):
                            lis.append(1/(1 + math.exp(-0.9*element[index])))
                            b=True
                                
                                        
                                        
            if index==(len(element)-1):
                    main.append(lis)
        return main
        
        




    def feedForward(self, X):                                                       #forward propogation through the network
                                                                                 
        self.z  = [[sum(a * b for a, b in zip(X_row, w1_col))                       #dot product of input layer and first set of weights(w1)
                        for w1_col in zip(*self.w1)] 
                                for X_row in X] 
                                                                   

        self.z2 = self.sigmoid(self.z)                                              


        self.z3 = [[sum(a * b for a, b in zip(z2_row, w2_col))                      #dot product of hidden layer (z2) and second set of weights (3x1)
                        for w2_col in zip(*self.w2)] 
                                for z2_row in self.z2] 

        
                                                                             

        output = self.sigmoid(self.z3)                                               

        return output                                                               #predicted x and y velocity for lander
        
    
    
    def backward(self, X, y, output):                                             #backward propogation through the network





        self.output_error=[]                                                    
        lis=[]
        for i in range(len(y)):
            for j in range(len(output[0])):
                lis.append(y[i][j] - output[i][j])                              #error in output
        i=0
        while i<len(lis):
            self.output_error.append(lis[i:i+len(y[0])])
            i+=len(y[0])
            
                
        

        
        










        
         
        self.output_delta = [[a*b*0.9 for a, b in zip(i, j)] for i, j in zip(self.output_error, self.sigmoid(output,deriv=True))]               #calculation of gradient

        
        self.t_w2 = [[self.w2[j][i] for j in range(len(self.w2))] for i in range(len(self.w2[0]))]                                               #transpose of w2

        self.z2_error = [[sum(a * b for a, b in zip(a_row, b_col))                                                                                                                              
                        for b_col in zip(*self.t_w2)] 
                                for a_row in self.output_delta] 

        
         
        self.z2_delta = [[a*b*0.9 for a, b in zip(i, j)] for i, j in zip(self.z2_error, self.sigmoid(self.z2,deriv=True))]  #applying derivative of sigmoid to z2  
                                                                                                                            #calculation of gradient
        


        self.t_X = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]                                           #transpose of X

        self.k = [[sum(a*b*0.1 for a, b in zip(a_row, b_col))  
                        for b_col in zip(*self.z2_delta)] 
                                for a_row in self.t_X]

        
        

        output_sum=[]
        lis=[]
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                lis.append(self.w1[i][j] + self.k[i][j]+0.9*self.k_p[i][j])                                     #adjusting first set of weights(input to hidden layer)
        i=0
        while i<len(lis):
            output_sum.append(lis[i:i+len(self.w1[0])])
            i+=len(self.w1[0])

        self.w1=output_sum                                                                                    #updating  weights w1


        self.k_p=self.k                                                                                        #storing previous delta weights for using in momentum parameter


        

        


        

        
        self.t_z2 = [[self.z2[j][i] for j in range(len(self.z2))] for i in range(len(self.z2[0]))]    #transpose of z2 
        
        self.l = [[sum(a*b*0.1 for a, b in zip(a_row, b_col))                                        
                        for b_col in zip(*self.output_delta)] 
                                for a_row in self.t_z2]

        
          

        output_sum=[]
        lis=[]
        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
                lis.append(self.l[i][j] + self.w2[i][j]+0.9*self.l_p[i][j])                          # adjusting second set (hidden layer to output layer) weights
        i=0
        t=1
        while i<len(lis):
            output_sum.append(lis[i:i+len(self.w2[0])])
            i+=len(self.w2[0])

        self.l_p=self.l                                                                             #storing previous delta weights                                                                                 
        self.w2=output_sum                                                                          #updating  weights w2

        
        


        
        
        

        


        
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)

     
        
NN = NeuralNetwork()


for i in range(10):                                                                 #trains the NN 1000 times
    NN.train(train_set_i, train_set_o)
    


add=0
for i,j in zip(validation_set_o,NN.feedForward(validation_set_i)):
    for k,l in zip(i,j):
        c=((k-l)**2)
        add=add+c
rm=add/(len(validation_set_o)*2)
rmse=rm**(1/2)
print('Rmse = ' + str(rmse))                                                          #Rmse for validation set
    


        


add=0
for i,j in zip(test_set_o,NN.feedForward(test_set_i)):
   for k,l in zip(i,j):
       c=((k-l)**2)
       add=add+c
rm=add/(len(test_set_o)*2)
rmse=rm**(1/2)
print('Rmse = ' + str(rmse))                                                        #Rmse for test set


