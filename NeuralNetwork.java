import java.io.*;
import java.util.StringTokenizer;

/**
 * The Neural Network class can construct a 3 layered feedforward neural network with any number of 
 * activation, hidden, and output nodes. It can either load weights for the network or generate random 
 * weights, storing the weights in two two dimensional arrays. The class contains methods for the 
 * activation function of the nodes, the sigmoid f(x) = 1/(1+e^-x), as well as the derivative of 
 * the activation function, which is f(x)(1-f(x)). The class can save weights to a file, or load weights
 * from a file. The network can either take in a set of inputs and return the output based on the current 
 * weights, or it can train using gradient descent and optimize the weights with respect to a training set.
 * @author Aaron Huang
 * @version 2.22.15
 *
 */
public class NeuralNetwork
{
   private static double[] activation;
   private static double[] hidden;
   private static double[] output;
   private static double[][] weightsKJ;
   private static double[][] weightsJI;
   private static double[][] inputLetters;
   private static double[][] theoretical;
   
   /**
    * This constructor creates a neural network with a given number of activation nodes and hidden nodes, 
    * and randomizes the weights for all the connections, within a specified range.
    * @param a is the number of activation nodes
    * @param h is the number of hidden nodes
    * @param r is the range within which the random weights will be generated
    */
   public NeuralNetwork(int a, int h, int o, double r)
   {
      activation = new double[a];
      hidden = new double[h];
      output = new double[o];
      weightsKJ = new double[a][h];
      weightsJI = new double[h][o];
      inputLetters = new double[52][10000];
      theoretical = new double[52][5];
      
      for(int i=0; i<h; i++)
      {
         for(int n=0; n<a; n++)
         {
            weightsKJ[n][i] = Math.random()*r - r/2.;         
         }
      }
      
      for(int i=0; i<o; i++)
      {
         for(int n=0; n<h; n++)
         {
            weightsJI[n][i] = Math.random()*r - r/2.;       
         }
      }
   }
   
   /**
    * This constructor creates a neural network with a given number of activation and hidden nodes, and
    * loads a two dimensional array of weights to be used for the activation to hidden connections, and
    * a one dimensional array for the weights of the hidden to output connections in the network.
    * @param a is the number of activation nodes
    * @param h is the number of hidden nodes
    * @param wKJ is a two dimensional array of weights for the activation-hidden connections
    * @param wJI is a one dimensional array of weights for the hidden-output connections
    */
   public NeuralNetwork(int a, int h, double[][] wKJ, double[][] wJI)
   {
      activation = new double[a];
      hidden = new double[h];
      weightsKJ = wKJ;
      weightsJI = wJI;
      
   }
   
   /**
    * f is the activation function used in the hidden layer and output layer nodes. The sigmoid function
    * 1/(1+e^-x) is used as the activation function
    * @param input is the double value for the sigmoid function to calculated at
    * @return the value of the sigmoid function with respect to the given input
    */
   public static double f(double input)
   {
      return 1./(1.+Math.exp(-input));
   }
   
   /**
    * df is the derivative of the activation function, which conveniently is f(x)(1-f(x)) where f(x) is the 
    * sigmoid function 
    * @param input is the double value for the derivative df to be calculated at
    * @return the double value of the derivative function at the input value
    */
   public static double df(double input)
   {
      double temp = f(input);
      
      return temp*(1.-temp);
   }
   
   /**
    * the saveWeights function will save all the weights in the network to a text file,
    * named weights.txt, with weights separated by a comma.
    * @throws IOException 
    */
   public static void saveWeights() throws IOException
   {
      String str = "";
      
      for(int h=0; h<hidden.length; h++)
      {
         for(int a=0; a<activation.length; a++)
         {
            str += weightsKJ[a][h]+",";
         }
         
      }
      
      for(int n=0; n<output.length; n++)
      {
         for(int h=0; h<hidden.length; h++)
         {
            str += weightsJI[h][n]+",";
         }
      }
      
      BufferedWriter writer = new BufferedWriter(new FileWriter("weights.txt"));
      writer.write(str);
      writer.close();
      
      return;
   }
   
   /**
    * the loadWeights function will set all the weights in the network to values
    * specified in a text file.
    * @param filename is the name of the file to be read from
    * @throws IOException
    */
   public static void loadWeights(String filename) throws IOException
   {
      BufferedReader reader = new BufferedReader(new FileReader(filename));
      String str = reader.readLine();
      reader.close();
      
      StringTokenizer st = new StringTokenizer(str, ",");
      
      for(int h=0; h<hidden.length; h++)
      {
         for(int a=0; a<activation.length; a++)
         {
            weightsKJ[a][h] = Double.parseDouble(st.nextToken());
         }
      }
      
      for(int n=0; n<output.length; n++)
      {
         for(int h=0; h<hidden.length; h++)
         {
            weightsJI[h][n] = Double.parseDouble(st.nextToken());
         }
      }
   }
   
   /**
    * 
    * @param filename
    * @throws Exception
    */
   public static void loadLetter(String filename, int count) throws Exception
   {
      BufferedReader reader = new BufferedReader(new FileReader(filename));
      String str = reader.readLine();
      reader.close();
      
      StringTokenizer st = new StringTokenizer(str, ",");
      
      for(int i=0; i<10000; i++)
      {
         inputLetters[count][i] = Double.parseDouble(st.nextToken());
      }
   }
   
   /**
    * 
    * @param count
    */
   public static void loadTheoretical()
   {
      for(int i=1; i<53; i++)
      {
         String b = Integer.toBinaryString(i);
         int len = b.length();
         for (int crap = 0; crap < 5 - len; crap++)
         {
            b = "0" + b;
         }
         for(int n=0; n<5; n++)
         {
            theoretical[i-1][n] = Integer.parseInt(b.substring(n,n+1));
         }
      }
   }
   
   /**
    * simulation will return the output of the neural network with its current set of weights
    * @param input is an array of inputs corresponding to the number of activation nodes
    * @return the value contained in the output node
    */
   public static double[] simulation(double[] input)
   {
      activation = input;
      double sumA = 0.0;
      double sumB = 0.0;

      for(int x = 0; x < output.length; x++)
      {
         for(int i = 0; i < hidden.length; i++)
         {
            for(int n = 0; n < activation.length; n++)
            {
               sumA += activation[n]*weightsKJ[n][i];          //calculate hidden layer
            }
            hidden[i] = f(sumA);
            sumB += weightsJI[i][x]*hidden[i];                 //calculate output layer
            sumA = 0;
         }
         output[x] = f(sumB);
         sumB = 0;
      }
      
      return output;
   }
   
   /**
    * train will train the neural network using gradient descent. The partial of the
    * error function with respect to each of the weights will be taken, and then each
    * of the weights will be updated by the negative of the partial with respect to that
    * weight multiplied by a learning factor lambda, which determines what step size is 
    * used when moving downhill in gradient descent. The method will return the total error, 
    * the squared sum of the local errors of the training sets.
    * @param input is a two dimensional array containing the sets of inputs for all training sets.
    *          i.e. In the case of the XOR network, there are 4 training set inputs, (0,0), (0,1), (1,0), (1,1).
    * @param error is the goal error to be trained to
    * @return the total error of neural network with the new trained weights.
    */
   public static double train(double[][]input, double[][] theoretical, double error)
   {
      double Et = 1.0;                                                        //total error
      
      double[][] E = new double[input.length][output.length];                 //Error of the outputs of each training set
      
      while(Math.abs(Et) > error)
      {
         for(int x=0; x<input.length; x++)                                    //loop over training sets
         {
            activation = input[x];                                               
            double sumA[] = new double[hidden.length];                        //Sum activation*weight
            double sumB[] = new double[output.length];                        //Sum hidden*weight
            double sumC = 0;                                                  //Sigma i for calculating partial kj

            for(int y = 0; y < output.length; y++)                            //simulation loop to find the sums
            {
               for(int i = 0; i < hidden.length; i++)
               {
                  for(int n = 0; n < activation.length; n++)
                  {
                     sumA[i] += activation[n]*weightsKJ[n][i];                  
                  }
                  hidden[i] = f(sumA[i]);
                  sumB[y] += weightsJI[i][y]*hidden[i];                          
               }
               output[y] = f(sumB[y]);
               
               E[x][y] = theoretical[x][y] - output[y];
            }
            
            double[][] partialKJ = 
                  new double[activation.length][hidden.length];               //partials with respect to each kj weight
            double[][] partialJI = new double[hidden.length][output.length];  //partials with respect to each ji weight
            
            for(int a=0; a<activation.length; a++)                            //loop to calculate kj partials
            {
               for(int b=0; b<hidden.length; b++) 
               {
                  for(int c=0; c<output.length; c++)
                  {
                     sumC += E[x][c]*df(sumB[c])*weightsJI[b][c];
                  }
                  partialKJ[a][b] = -activation[a]*df(sumA[b])*sumC;          //-ak*f'(k(ak*wkj))*i(Ei*f'(j(hj*wji))*wji)
                  sumC = 0;
               }
            }
            
            for(int e=0; e<hidden.length; e++)                                //loop to calculate ji Partials
            {
               for(int f=0; f<output.length; f++) 
               {
                  partialJI[e][f] = -E[x][f]*df(sumB[f])*hidden[e];           //-(Ei*f'(j(hj*wji)hj))
               }
            }
            
            double lambda = 1.;                                               //learning factor

            for(int h=0; h<hidden.length; h++)                                //adapt weights with gradient descent
            {
               for(int a=0; a<activation.length; a++)                         //adapt kj weights
               {
                  weightsKJ[a][h] += -lambda*partialKJ[a][h];
               }
               
               for(int o=0; o<output.length; o++)                             //adapt ji weights
               {
                  weightsJI[h][o] += -lambda*partialJI[h][o];
               }
            }
            
         }                                                                    //end for(int x=0; x<input.length; x++)
         
         Et = 0.0;
         for(int i=0; i<input.length; i++)                                    //loop over training sets to find total error
         {
            for(double d: E[i])
            {
               Et += d*d;
            }
         }
         Et = Math.sqrt(Et);
         
         System.out.println("error: " + Et);
      }                                                                       //end while(Math.abs(Et) > error)
      
      return Et;
   }
   
   /**
    * the main method will create a neural network with 2 activation nodes and 3 hidden nodes,
    * and train the network to solve the XOR problem
    * @param args
    */
   public static void main(String[] args) throws Exception
   {
      //double[][] initialWeightKJ = {{67.14, 46.64, -56.08},{95.28, -4.05, 92.80}};
      //double[] initialWeightJI = {93.16, -64.53, -44.31};
      //new NeuralNetwork(2,3,initialWeightKJ, initialWeightJI);
      
      new NeuralNetwork(10000,30,5,.5);
      
      //DibDump d = new DibDump();
      
      loadLetter("a.txt", 0);
      loadLetter("b.txt", 1);
      loadLetter("c.txt", 2);
      loadLetter("d.txt", 3);
      loadLetter("e.txt", 4);
      loadLetter("f.txt", 5);
      loadLetter("g.txt", 6);
      loadLetter("h.txt", 7);
      loadLetter("i.txt", 8);
      loadLetter("j.txt", 9);
      loadLetter("k.txt", 10);
      loadLetter("l.txt", 11);
      loadLetter("m.txt", 12);
      loadLetter("n.txt", 13);
      loadLetter("o.txt", 14);
      loadLetter("p.txt", 15);
      loadLetter("q.txt", 16);
      loadLetter("r.txt", 17);
      loadLetter("s.txt", 18);
      loadLetter("t.txt", 19);
      loadLetter("u.txt", 20);
      loadLetter("v.txt", 21);
      loadLetter("w.txt", 22);
      loadLetter("x.txt", 23);
      loadLetter("y.txt", 24);
      loadLetter("z.txt", 25);
      loadLetter("au.txt", 26);
      loadLetter("bu.txt", 27);
      loadLetter("cu.txt", 28);
      loadLetter("du.txt", 29);
      loadLetter("eu.txt", 30);
      loadLetter("fu.txt", 31);
      loadLetter("gu.txt", 32);
      loadLetter("hu.txt", 33);
      loadLetter("iu.txt", 34);
      loadLetter("ju.txt", 35);
      loadLetter("ku.txt", 36);
      loadLetter("lu.txt", 37);
      loadLetter("mu.txt", 38);
      loadLetter("nu.txt", 39);
      loadLetter("ou.txt", 40);
      loadLetter("pu.txt", 41);
      loadLetter("qu.txt", 42);
      loadLetter("ru.txt", 43);
      loadLetter("su.txt", 44);
      loadLetter("tu.txt", 45);
      loadLetter("uu.txt", 46);
      loadLetter("vu.txt", 47);
      loadLetter("wu.txt", 48);
      loadLetter("xu.txt", 49);
      loadLetter("yu.txt", 50);
      loadLetter("zu.txt", 51);
      
      loadTheoretical();
      
      train(inputLetters,theoretical, 0.1);
      System.out.println("trained");
      saveWeights();
      System.out.println("saved");
      
      loadWeights("weights.txt");
      System.out.println("a: " + simulation(inputLetters[0])[0]);

      return;
   }
}
