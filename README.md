# Movie Recommendation System

This project implements a **collaborative filtering-based recommendation system** using K-Nearest Neighbors (KNN) to suggest movies to users based on their previous ratings. The dataset used consists of movie ratings provided by users, and the recommendation system predicts movies that users are likely to enjoy.

## Datasets
Two datasets are used for this project:
- **Movies Dataset** (`movie.csv`): Contains movie information like `movieId` and `title`.
- **Ratings Dataset** (`rating.csv`): Contains user ratings of movies, including `userId`, `movieId`, and `rating`.

### Features:
- **userId**: Unique identifier for each user.
- **movieId**: Unique identifier for each movie.
- **rating**: Rating given by a user to a movie, ranging from 0.5 to 5.0.
- **title**: Title of the movie.

## Key Steps:
1. **Data Preprocessing**:
   - The data is merged on `movieId` to combine user ratings with movie titles.
   - Missing values are checked and removed if necessary.

2. **Exploratory Data Analysis (EDA)**:
   - **Distribution of Ratings**: A histogram is plotted to visualize how ratings are distributed.
   - **Ratings per Movie and User**: Histograms are plotted to analyze the number of ratings per movie and per user, with log scaling for better visualization.

3. **Collaborative Filtering**:
   - A **pivot table** is created with movies as rows and users as columns. Each cell represents the rating given by a user to a movie.
   - The user-movie matrix is converted into a sparse matrix for efficient computation.
   - **K-Nearest Neighbors (KNN)** is used to compute distances between movies based on user ratings, and similar movies are recommended.

4. **Model Evaluation**:
   - **Root Mean Squared Error (RMSE)** is calculated to evaluate the accuracy of predicted ratings.
   - **Precision, Recall, and Accuracy**: The ratings are binarized (relevant if rating >= 4.0) to calculate precision, recall, and accuracy.
   - **Receiver Operating Characteristic (ROC) and Area Under Curve (AUC)** are computed to assess the performance of the recommendation system.

5. **Visualization**:
   - Scatter plot comparing actual and predicted ratings.
   - Bar plot showing precision and recall scores.
   - ROC curve to visualize the trade-off between true positive rate and false positive rate.

## Libraries Used:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`
- `mlxtend`



## Metrics:
- **RMSE**: Root Mean Squared Error evaluates the prediction error for ratings.
- **Precision & Recall**: Evaluates how well the system identifies relevant movie recommendations.
- **AUC**: Area Under the Curve provides an overall measure of the recommendation system’s performance.

## Results:
- **Recommendations**: The system provides a list of movies similar to a chosen movie based on user ratings.
- **Performance**: The system’s performance is evaluated using RMSE, Precision, Recall, and ROC-AUC metrics.

## References:
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
