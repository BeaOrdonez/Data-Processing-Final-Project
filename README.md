# Data-Processing-Final-Project

Here we describe the development of each step of the project. The main objective is to apply Natural Language Processing (NLP) to process a set of recipies in the dataset "full_format_recipes". 

1. ANALYSIS OF INPUT VARIABLES

    First of all, our first step was to import the corresponding libraries and modules needed for our Python program. Those are the pandas library in order 
 to work with structured data, NumPy library for numerical computations and working with arrays, the termcolor library used to colorize text printed in the 
 terminal, the seaborn and matplotlib library for creating statistical visualizations and more libraries that will be mention along the explication of our 
 project and code.
 
    Secondly, we import the database mentioned above and we analyze the input data, which is a diccionary of 20130 cook recipes. Each recipe is define 
 with different variables, in total 8 variables that describe each recipe. These variables are 'directions', 'fat', 'date', 'categories', 'calories', 
 'description', 'protein', 'rating', 'title', 'ingredients' and 'sodium'. The variables 'fat', 'date', 'calories', 'protein', 'rating' and 'sodium' 
 contain numerical values and the rest of the variables contain text.
    
    As a third step, we eliminate null lines in this dataset and proceed to visualize the relationship between the different input 
 variables. We focuss on the 'rating' of each recipe as our target variable and it's relation with the 'categories'. We look to see if the rating of a 
 recipe depends on the categorie it has. Firstly we keep the 10 most common categories among all recipies, using functions like 'value_counts()' and 'explode()'
 which help us isolate the categories. Then we look for the ratings of each of them and we plot a figure that represents the average ratings per category.
 Also, as a complementary step, we tested the correlation between numerical variables such as fat and calories with the target rating. We computed
 a correlation matrix where a positive correlation closer to 1 between 2 variables means that they have a proportional relation, if one increases, the other one increases as well.
 Correlation close to 0 means that there is no linear relation and negative correlation means that the variables are inversely related. The matrix can be seen in the following picture:

 ![image](https://github.com/user-attachments/assets/70e7ed09-cbdd-4200-be20-815723f3652c)


2. IMPLEMENTATION OF A PIPELINE FOR TEXT PROCESSING

     In this section we first import the NLTK library, which is a powerful toolkit for working with text data in Python. Then we also import the re module, used for pattern matching and text cleaning.
  We use preprocessing functions in order to improve the quality of textual data by eliminating elements that do
contribute to the analysis, unifying the format of the text and reducing the complexity of
  the vocabulary, in order to prepare the text for models to process it more efficiently and accurately.

   2.1. Preprocess "Desc" column
   
      In order to process the descriptions of our recipies we have created a function called "preprocess_text" which takes each input text and converts it to non capital letters and removes
   special characters, numbers and stopwords. This function is applied to our data. Then, we tokenize the text using the function "tokenize". The output is a dictionary called "my corpus".
   This dictionary contains 5764 terms, words present in the description variable of all the recipes in the dataset.

   ![image](https://github.com/user-attachments/assets/a42e5239-b445-42b4-bd42-95619402516a) 

   2.2. Identify the number of descriptions in which each token appears
   
      In this step we identify the frequency of each token, the number of recipies in which each token appears. The output shows two columns (Token, Number of descriptions where it appears).
   We have programmed the code to do so in the following way:
   
   - Build a DataFrame with each token and its frequency in the descriptions.
   - Sort tokens by frequency.
   - Count and filter the tokens that are infrequent or very frequent, filter extremes.
   - Visualize the logarithmic distribution of the frequencies with a histogram.
     Printing the following result:
     
     ![image](https://github.com/user-attachments/assets/7b902a8b-48ca-42da-aaa6-ac0df5e3687a)
   
   
   2.3. Bag of words representation of the corpus

   Bag of words is a simple way to convert text into numerical data. We create a vocabulary of unique words (mycorpus) from the entire corpus (all descriptions of the recipes).
   For each description, count how many times each word from the vocabulary appears. The code programmed to do so, returns a sparse vector, which is a list of tuples
   (token_id, frequency). Where token_id is the ID of the word in the dictionary D and frequency is how many times that word appears in the recipe.

4. VECTOR REPRESENTATION OF THE DOCUMENTS USING THREE DIFFERENT PROCEDURES
   
     In this section three vectorization methods have been used, TF-IDF, Word2Vec and Trasformers.
   It is important to mention that the input data used here corresponds only to the column "Descriptions".

   --> TF - IDF

       To develop this method, BoW (Bag of words) has been previously done. As it has been sais, BoW provides what we call corpus, which is the base of our TF-IDF model.
       The data used here has been previously treated using the preprocessed function mentioned above.
       In this section, the main objetive is to transform the corpus in to a weighted representation based not only on the frequency of words in a recipe but also
       and across the whole dataset. to do so, Gensim libary has been used. The output shows tuples containing (word_id, tfidf_value). Each recipe has a TF IDF vector
       which contains as many tuples as tokens in the description.

   --> Word2vec

       For Word2vec vectorization we also use Gensim library to create word embeddings using the previous tokenized corpus. Each description is shown as the mean of the
       embeddings of it's words. Words that are not in the vocubulary of the model are not considered. Results are saved in a numpy array. The produce is the following,
       considering the chosen parameters. We feed the model with our previous BoW corpus, we set the size of the generated vectors to 100 (vector_size=100), we consider
       a context of 5 words, before and after the target word (window=5) and we inlcude all the words in the corpus that appear at least one time (min_count=1). Finally
       we represent each description as the average of word embeddings using the function "Average_word2vec".
       
   --> Transformers

       This is a much more complex method which produces the embeddings according to the context of each word. RobertaModel is used for this purpose. Texts are tokenized
       ensuring a maximum input leght of 512 tokens. Then data is processed in batches of size 16 for more efficient computation. We obtain the mean embeddings from the
       last hidden layer, obtaining one vector representation per input, which is stored in a numpy array. The number of hidden layers and attention heads used is set by
       the roberta-based model.
   
       -- 1. Importing necessary tools
           - datasets: A Hugging Face library to load and process datasets.
           - torch: PyTorch, used for handling tensors and running models like RoBERTa.
   
       -- 2. Transformer configuration
            In this step, the purpose is to extract semantic embeddings from the recipe's descriptions using a pre-trained transformer model (RoBERTa). These embeddings
            capture the meaning of the descriptions in a numerical form that machine learning models can use.
   
       -- 3. Apply to our data
            In this section, we applied the tools prepared in the step before to process the dataset and generate embeddings. To do so:
          - First, we filter invalid data: Keeps only valid descriptions.
          - Generate embeddings: Processes the dataset in manageable batches, using the RoBERTa model to extract embeddings for each recipe description.
          - Save embeddings: Converts embeddings into a NumPy array and stores them for efficient reuse.
            After this has been completed, we have a clean dataset and a corresponding embedding file (desc_embeddings.npy) that represents each recipe description as a
            dense vector, ready for downstream machine learning or analysis tasks later on.
   
       -- 4. Use of PCA for dimensionality reduction
           As the title says, here we reduce the embedding size (768 dimensions) to 100 dimensions for efficiency. 100 dimension that we believe that represent most of the variance
           of the input data. 
   
       -- 5. Load saved embeddings
           Loading the previously saved embeddings (from Step 3, the original embeddings (desc_embeddings.npy) and the reduced embeddings (desc_embeddings_pca.npy))
           into memory, so they can be used in further analysis or machine learning tasks.
           In addition, we also extract the ratings column from the dataset as the target variable for training machine learning models.


5. TRAINING AND EVALUATION OF REGRESSION MODELS

   Once the vectorization of the descriptions has been done we proceed to use the embeddings to train and evaluate predicction models. Random Forest and Neural Networks have been chosen. The scikit learn tool and pytorch were used. 

   For the implementation of the Neural Network, here are the followed steps and their hyperparameters:  

   --> Data splitting step
    
    In data splitting, test_size = 0.2 to put 20% of the data in the testing dataset to ensure that the model is evaluated on unseen data.  
    
   --> Model implementation

    ReLU Activation Function helps the network learn efficiently by adding non-linearity so it effects learning efficiency and non-linearity
    Number of hidden layers and neurons determines the model's capacity to learn patterns. 128 has been chosen for the first layer neuron number and 64 for the second one. 

   --> Training step

    Adam optimizer, efficient for deep learning tasks, has been used to adjust the model's weights using gradients to minimize the loss function. The learning rate controls the size of updates to the weights. Too
    high value can overshoot the optimal solution and too low value slows training, so we chose 0.001 for our case. The number of epochs is the number of times the model sees the training data and the batch size is
    the number of data processed at a time for efficiency. 

   --> In the evaluation step, the following metrics have been used.

    Mean Squared error or test Loss (MSE): Indicates how well the model predicts unseen data. It measures the average squared difference between predicted and actual values. The smaller the value is, the better predictions are.
    R² Score: measures how well the model explains the variance in the target variable. The closer the value is to 1, the better it is.
    Improvements: With optimized hyperparameters and sufficient data preprocessing, this approach could provide better results. More features may improve results but may  require careful preprocessing to avoid noise. We could also
    consider the data normalization. 

   Implementation of the Random Forest:
 
    Thanks to Scikit Learn, this implementation is really easy to do. To get our results, we only needed two hyper parameters for the data splitting part and as in the Neural Network, we used 20% of the data for testing and
    80% for the training part. This 80% is used to train the RandomForestRegressor model provided by scikit learn. Then the model predictions are made on the test data. Some hyperparameters such as random_state with the default
    value 42, n_estimators commonly set at 100 could be used to improve the results by      trying to tune them to better values for better results. The same metrics of evaluation MSE and R² Score have been used. 
    
    The results are represented in the table below. Here the two models have been used with they default parameters as defined in the explanations. Nevertheless, in order to achieve better results, GridSearchCV has been tried
    for Random Forest to perform Hyperparameter selection, with results that do not differ much from the ones obtain with the default parameters.

![image](https://github.com/user-attachments/assets/97f8cf61-5410-4b86-a7d9-a26c3caa4885)

      In this table can clearly be seen how the results are not as expected. R2 score and Mean Square Error (mse) have been chosen to measure the ability of the models according the different input
  vecotrization methods. Focussing on the well known mse, we can appreciate that the best result is obtained for Random forest with TF-IDF while the worst is the Neural Network with the same
  TF-IDF vector. With optimized hyperparameters and bigger input data size, these evaluations could provide better results. More features may improve results but may  require careful preprocessing to avoid noise. We could also consider the data normalization. 

5. FINE-TUNNING

   As a final attempt to improve the results, a transformer with a regression head has been implemented here. As a first step we need to train the model. The procedure is the following.
   
   --> We split out dataset into two train and test smaller datasets.
   --> We tokenized the training data
   --> Convert it to tensors and create a tensorDataset which we use to train the model using 3 epochs (Three complete passes through the entire dataset)
   --> As a last step we prepare the test data the same way and we use it to test the predicting performance of the model. The results are shown below.

![image](https://github.com/user-attachments/assets/11efd02e-a7f1-4e57-8bee-63e66d9aa8f5)

    It can clearly be seen that the results are not as expected. The Mean Square Error is 1.73 and the R2 score does not even reach 0.1. 
    
6. EXTENSION: BYTE-PAIR ENCODING (BPE) 

For the extension of our project, we chose another data representation named BPE (Byte-Pair Encoding).
To train the BPE tokenizer, it has been initialized and configured with:

-->vocab_size = 5000 which limits the vocabulary to 5000 most frequent subwords.

-->min_frequency = 2 that includes only tokens that appear at least twice.

-->Special tokens ([PAD], [UNK], etc.) are added for padding and handling unknown words.

Then, the tokenizer is trained on our descriptions list and saved for reuse later.
In the encode description, each description is tokenized into subword IDs using the trained tokenizer saved previously. The result is a list of sequences of varying lengths, as each description is tokenized into different numbers of subword IDs.
To evaluate with Random Forest, since this algorithm requires fixed-length input, each sequence is padded to the length of the longest sequence in the dataset.
Padding is done using 0 (commonly used as a [PAD] token). 
As a result, we got Mean Squared Error: 1.5551816774332363, R² Score:0.08452808785616595

7. COMBINATION OF THREE FEATURES: directions, descriptions and categories in a unique dataset as the input variables

In this part, we combined the three features in one input data for each recipe and repete the process to get the following results with random forest:

-->Word2Vec:  Mean Squared Error: 1.5661794699829648, R² Score: 0.07805413672816763


-->TD - IDF: Mean Squared Error: 1.6592607855772863, R² Score: 0.023260969339108373


It can be clearly seen that Word2Vec is the best one with the minimum MSE and the closiest R2 score to 1. 
