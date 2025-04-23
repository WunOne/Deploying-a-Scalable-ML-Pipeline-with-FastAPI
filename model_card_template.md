# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is using the RandomForestClassifier model on a Census Bureau dataset.

## Intended Use
This ML model is designed to make predictions on salary based upon factors such as age, 
workclass, education, race, sex, etc.

## Training Data
Two thirds of the dataset from the Census Bureau was subset for training the model.

## Evaluation Data
The remaining one third of the dataset from the Census Bureau was used for testing the model.

## Metrics
Precision: 0.7316 | Recall: 0.6281 | F1: 0.6759

## Ethical Considerations
There are several ethical considerations in this data; the most obvious is historical bias. 
The dataset is from the 1994 United States Census and is not representative of modern 
demographics and labor markets.

## Caveats and Recommendations
I would caveat this dataset as contextual. First, the dataset is only a slice of the 1994 U.S. Census, and the model makes predictions from incomplete data. Second, there are many factors as to why an individual would make more or less than a given salary, many of which are not represented in the dataset, and are, in fact, more qualitative and could make it more difficult to analyze. This could include factors such as someone being on short-term or long-term disability and making less than their regularly earned salary, or a nationwide financial crisis, where several people lost employment, reducing the income earned that year.