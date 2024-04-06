# -IndustrialCopperModeling
This project aims to develop a machine learning tool to analyze copper sales data and predict two things: future selling prices and likelihood of winning a sale (Won/Lost). It will use Python libraries and culminate in a user-friendly web interface.

Introduction:
Enhance your proficiency in data analysis and machine learning with our "Industrial Copper Modeling" project. In the copper industry, dealing with complex sales and pricing data can be challenging. Our solution employs advanced machine learning techniques to address these challenges, offering regression models for precise pricing predictions and lead classification for better customer targeting. You'll also gain experience in data preprocessing, feature engineering, and web application development using Streamlit, equipping you to solve real-world problems in manufacturing.

Key Technologies and Skills:
*Python
*Numpy
*Pandas
*Scikit-Learn
*Matplotlib
*Seaborn
*Pickle
*Streamlit

Installation:-
To run this project, you need to install the following packages:
pip install numpy
pip install pandas
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
pip install streamlit

#Features :-
Data Preprocessing:
*Data Understanding: Before diving into modeling, it's crucial to gain a deep understanding of your dataset. Start by identifying the types of variables within it, distinguishing between continuous and categorical variables, and examining their distributions. In our dataset, there might be some unwanted values in the 'Material_Ref' feature that start with '00000.' These values should be converted to null for better data integrity.
*Handling Null Values: The dataset may contain missing values that need to be addressed. The choice of handling these null values, whether through mean, median, or mode imputation, depends on the nature of the data and the specific feature.
*Encoding and Data Type Conversion: To prepare categorical features for modeling, we employ ordinal encoding. This technique transforms categorical values into numerical representations based on their intrinsic nature and their relationship with the target variable. Additionally, it's essential to convert data types to ensure they match the requirements of our modeling process.
*Skewness - Feature Scaling: Skewness is a common challenge in datasets. Identifying skewness in the data is essential, and appropriate data transformations must be applied to mitigate it. One widely-used method is the log transformation, which is particularly effective in addressing high skewness in continuous variables. This transformation helps achieve a more balanced and normally-distributed dataset, which is often a prerequisite for many machine learning algorithms.
*Outliers Handling: Outliers can significantly impact model performance. We tackle outliers in our data by using the Interquartile Range (IQR) method. This method involves identifying data points that fall outside the IQR boundaries and then converting them to values that are more in line with the rest of the data. This step aids in producing a more robust and accurate model.
*Wrong Date Handling: In cases where some delivery dates are precedes the item dates, we resolve this issue by calculating the difference and it's used to train a Random Forest Regressor model, which enables us to predict the corrected delivery date. This approach ensures that our dataset maintains data integrity and accuracy.

Exploratory Data Analysis (EDA) and Feature Engineering:
*Skewness Visualization: To enhance data distribution uniformity, we visualize and correct skewness in continuous variables using Seaborn's Histplot and Violinplot. By applying the Log Transformation method, we achieve improved balance and normal distribution, while ensuring data integrity.

*Outlier Visualization: We identify and rectify outliers by leveraging Seaborn's Boxplot. This straightforward visualization aids in pinpointing outlier-rich features. Our chosen remedy is the Interquartile Range (IQR) method, which brings outlier data points into alignment with the rest of the dataset, bolstering its resilience.

*Feature Improvement: Our focus is on improving our dataset for more effective modeling. We achieve this by creating new features to gain deeper insights from the data while making the dataset more efficient. Notably, our evaluation, facilitated by Seaborn's Heatmap, confirms that no columns exhibit strong correlation, with the highest correlation value at just 0.42 (absolute value), underlining our commitment to data quality and affirming that there's no need to drop any columns.

Classification:
*Success and Failure Classification: In our predictive journey, we utilize the 'status' variable, defining 'Won' as Success and 'Lost' as Failure. Data points with status values other than 'Won' and 'Lost' are excluded from our dataset to focus on the core classification task.

*Handling Data Imbalance: In our predictive analysis, we encountered data imbalance within the 'status' feature. To address this issue, we implemented the SMOTETomek oversampling method, ensuring our dataset is well-balanced. This enhancement significantly enhances the performance and reliability of our classification tasks, yielding more accurate results in distinguishing between success and failure.

*Algorithm Assessment: In the realm of classification, our primary objective is to predict the categorical variable of status. The dataset is thoughtfully divided into training and testing subsets, setting the stage for our classification endeavor. We apply various algorithms to assess their performance and select the most suitable base algorithm for our specific data.

*Algorithm Selection: After thorough evaluation, two contenders, the Extra Trees Classifier and Random Forest Classifier, demonstrate commendable testing accuracy. Upon checking for any overfitting issues in both training and testing, both models exhibit strong performance without overfitting concerns. I choose the Random Forest Classifier for its ability to strike a balance between interpretability and accuracy, ensuring robust performance on unseen data.

*Hyperparameter Tuning with GridSearchCV and Cross-Validation: To fine-tune our model and mitigate overfitting, we employ GridSearchCV with cross-validation for hyperparameter tuning. This function allows us to systematically explore multiple parameter values and return the optimal set of parameters. {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2}

*Model Accuracy and Metrics: With the optimized parameters, our Random Forest Classifier achieves an impressive 96.5% accuracy, ensuring robust predictions for unseen data. To further evaluate our model, we leverage key metrics such as the confusion matrix, precision, recall, F1-score, AUC, and ROC curve, providing a comprehensive view of its performance.

*Model Persistence: We conclude this phase by saving our well-trained model to a pickle file. This enables us to effortlessly load the model and make predictions on the status whenever needed, streamlining future applications.

Regression:
*Algorithm Assessment: In the realm of regression, our primary objective is to predict the continuous variable of selling price. Our journey begins by splitting the dataset into training and testing subsets. We systematically apply various algorithms, evaluating them based on training and testing accuracy using the R2 (R-squared) metric, which signifies the coefficient of determination. This process allows us to identify the most suitable base algorithm tailored to our specific data.

*Algorithm Selection: After thorough evaluation, two contenders, the Extra Trees Regressor and Random Forest Regressor, demonstrate commendable testing accuracy. Upon checking for any overfitting issues in both training and testing, both models exhibit strong performance without overfitting concerns. I choose the Random Forest Regressor for its ability to strike a balance between interpretability and accuracy, ensuring robust performance on unseen data.

*Hyperparameter Tuning with GridSearchCV and Cross-Validation: To fine-tune our model and mitigate overfitting, we employ GridSearchCV with cross-validation for hyperparameter tuning. This function allows us to systematically explore multiple parameter values and return the optimal set of parameters. {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}.

*Model Accuracy and Metrics: With the optimized parameters, our Random Forest Regressor achieves an impressive 95.7% accuracy. This level of accuracy ensures robust predictions for unseen data. We further evaluate our model using essential metrics such as mean absolute error, mean squared error, root mean squared error, and the coefficient of determination (R-squared). These metrics provide a comprehensive assessment of our model's performance.

*Model Persistence: We conclude this phase by saving our well-trained model to a pickle file. This strategic move enables us to effortlessly load the model whenever needed, streamlining the process of making predictions on selling prices in future applications.

#The Workflow of the code is:
Web App Setup:
Import necessary libraries (date, NumPy, Pickle, Streamlit).
Define custom Streamlit page configurations (title, background color removal).
Create custom CSS styles for submit button and prediction results.
    
User Input Options:
Establish various options for user selection (countries, status values, etc.).
Create dictionaries to convert these options into numerical values for models.
                                                                  
Prediction Functions:
Define functions for both regression (price prediction) and classification (sales status prediction).
                                                                  
Regression Function:
Get user input for features like date, quantity, country, etc.
Load the pre-trained regression model using Pickle.
Convert user input into a NumPy array suitable for model input.
Perform prediction using the loaded model.
Apply inverse transformation for log-transformed data (quantity & thickness).
Round the predicted selling price and display it with custom styling.
             
Classification Function:
Similar to regression, get user input and load the classification model.
Prepare user input data in a format compatible with the model.
Perform prediction using the loaded model and retrieve the predicted status (Won/Lost).
Display the predicted status ("Won" or "Lost") with custom styling.

Web App Interface:
Create tabs for "PREDICT SELLING PRICE" and "PREDICT STATUS".
Based on the selected tab, call the corresponding prediction function (regression or classification).
Handle potential errors (missing user input) and display warning messages.
Display the predicted selling price or status with custom styling and success/failure indicators.
             
Error Handling:
Implement error handling mechanisms to catch missing user input.
Display informative warning messages for empty fields
                   
Presentation:
Use custom CSS styles to enhance the visual appeal of the submit button and prediction results.
                   
Model Loading (Implicit):
The code relies on pre-trained models saved using Pickle (regression_model.pkl and classification_model.pkl).
These models are assumed to be available for loading during application execution.
                                        
Data Preprocessing (Implicit):
The code assumes the user input data is already preprocessed in a way that aligns with how the models were trained.
This preprocessing might involve handling missing values, encoding categorical variables, etc.
                                        
Model Selection (Implicit):
Based on the chosen tab (price prediction or status prediction), the appropriate model is loaded and used for inference.
                                        
User Interaction:
The web app allows users to easily interact with the interface by providing input through forms.
The submit button triggers the prediction process based on the selected tab.
                                        
Prediction Output:
The predicted selling price or status is displayed prominently on the web interface.
Custom styling is applied to enhance readability and provide visual cues on success (green) or failure (red).

Real-Time Functionality (Implicit):
While the application itself doesn't predict in real-time (uses historical data), trained models have the potential to provide real-time benefits for price estimation and sales success prediction based on new input data.

Scalability (Implicit):
The web app can potentially scale to handle multiple users by leveraging Streamlit's capabilities.

Deployment (Implicit):
This code snippet focuses on the application logic. Deployment on a server would be required to make it accessible over the internet.
                                        
Maintenance:
The application might require ongoing maintenance to ensure the models stay up-to-date with new data and maintain prediction accuracy.
                                        
Security (Implicit):
Security considerations are crucial for real-world deployments, especially if the application handles sensitive data.
                                                              
Documentation (Implicit):
Proper documentation of the code, models, and user instructions is essential for future reference and maintainability.
                                                              
Further Enhancements:
The application can be extended with features like data visualization for user insights, input validation for data quality control, and integration with external data sources.

#real time examples of the project:-
While this specific code snippet deals with pre-trained models and doesn't deliver predictions in real-time, the underlying concept can be applied in real-time scenarios with some modifications. Here are real-time examples of how this type of project can be used in the copper industry:

1. Real-time Price Estimation during Negotiations:

A salesperson can use the application during customer negotiations.
By entering real-time details like quantity, delivery date, and customer information, the salesperson can get an instant predicted selling price.
This allows for more informed pricing decisions and potentially leads to quicker deal closures and more competitive offers.
2. Dynamic Inventory Management:

The application can be integrated with real-time sales data and inventory management systems.
Based on predicted prices and sales likelihood from the model, businesses can optimize inventory levels in real-time.
This helps reduce the risk of stockouts or holding excess inventory, leading to improved cash flow and resource allocation.
3. Real-time Sales Pipeline Monitoring:

The application can be integrated with a CRM system to analyze real-time sales data.
By feeding this data into the classification model (predicting Won/Lost), businesses can gain insights into the health of their sales pipeline.
This allows for real-time identification of high-risk deals and enables focused efforts on improving conversion rates.
4. Market Trend Analysis with Real-time Data:

The application can be designed to ingest real-time market data on copper prices, competitor activity, and economic indicators.
By feeding this data into the model (regression or classification), businesses can gain insights into real-time market trends and customer preferences.
This allows for quicker adaptation of strategies and product offerings to capitalize on emerging market opportunities.

#The real-time benefits of this copper modeling project lie in its ability to leverage constantly updating information to inform business decisions. Here's a breakdown of the benefits mentioned in the passage:

Faster and More Competitive Deals: Salespeople can get instant price estimations based on real-time factors, leading to quicker deal closures and potentially more attractive offers.
Optimized Inventory Management: Businesses can adjust inventory levels in real-time based on predicted prices and sales likelihood, reducing stockouts and excess inventory, which improves cash flow and resource allocation.
Improved Sales Pipeline Monitoring: Real-time analysis of sales data helps identify deals at risk of falling through, allowing for focused efforts to improve conversion rates and boost overall sales success.
Actionable Market Insights: By ingesting real-time market data, the project provides insights into current trends and customer preferences. This allows businesses to adapt their strategies and product offerings quickly to capitalize on emerging market opportunities and stay ahead of the competition.
In essence, the real-time benefits translate to increased efficiency, better decision-making, and a significant competitive advantage in the copper industry.
Key Considerations for Real-time Implementation:

Data Streaming: Integrating the application with real-time data sources like sales dashboards, market feeds, and CRM systems.
Model Updates: Regularly retraining the models with new data to ensure prediction accuracy reflects real-time market conditions.
Latency Management: Optimizing data pipelines and model inference to minimize delays between data acquisition and prediction delivery.
By implementing these real-time elements, this type of copper modeling project can become a valuable tool for businesses to make data-driven decisions, optimize sales strategies, and gain a competitive edge in the market.
             
