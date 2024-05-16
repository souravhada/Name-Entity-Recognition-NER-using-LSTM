
# Named Entity Recognition (NER) using LSTM

## Overview

### Named Entity Recognition (NER)
Named Entity Recognition (NER) is a subtask of natural language processing (NLP) focused on identifying and categorizing key information (entities) in text into predefined categories. Common categories include person names, organizations, locations, date expressions, quantities, monetary values, and percentages. NER is crucial for various applications such as information retrieval, question answering systems, content classification, and knowledge graph construction.

### Long Short-Term Memory Networks (LSTMs)
Long Short-Term Memory networks (LSTMs) are a special kind of Recurrent Neural Network (RNN) capable of learning long-term dependencies. LSTMs are designed to avoid the long-term dependency problem by using a series of gates that control the flow of information. These include:
- **Input gate**: Determines how much of the new input to let into the cell state.
- **Forget gate**: Decides what information is discarded from the cell state.
- **Output gate**: Determines what information to output based on the current input and the memory of the cell.

LSTMs are particularly useful in tasks where context and sequence in data are essential, making them ideal for applications like time series prediction, sequence prediction, and natural language processing tasks, including NER.

### CoNLL-2003 Dataset
The CoNLL-2003 dataset is a benchmark dataset widely used for training and evaluating NER systems. It originated from the CoNLL-2003 shared task and includes text from Reuters news stories, annotated for named entity recognition. The dataset is notable for its four types of named entities: persons (PER), organizations (ORG), locations (LOC), and miscellaneous (MISC).

## Data Preprocessing
- **Loading the Dataset**: The dataset is loaded and prepared for training.
- **Tokenization**: Text data is tokenized to convert sentences into sequences of words.
- **Label Encoding**: Named entities are encoded into numerical labels for model training.

## Model Building
- **LSTM Model Architecture**: The LSTM model is built using Keras with layers including Embedding, Bidirectional LSTM, and Dense layers.
- **Compilation**: The model is compiled with appropriate loss functions and optimizers.

## Training the Model
- **Training Process**: The model is trained on the training dataset with validation on a separate validation set.
- **Hyperparameter Tuning**: Hyperparameters such as batch size, learning rate, and number of epochs are tuned for optimal performance.

## Evaluation and Results
- **Evaluation Metrics**: The model is evaluated using metrics like accuracy, precision, recall, and F1-score.
- **Results**: The performance of the model on the test dataset is reported with detailed results and analysis.

## Installation

To run this notebook, you need the following packages:

```bash
pip install tensorflow numpy pandas seqeval
```

## Usage

Run the notebook cells sequentially to train and evaluate the NER model. Make sure to have the CoNLL-2003 dataset available in the specified path.

## Conclusion

This project demonstrates the use of LSTM networks for the task of Named Entity Recognition (NER) using the CoNLL-2003 dataset. The model's performance indicates its effectiveness in identifying and categorizing named entities in text.
