Question Bank Analysis

This project analyzes a set of questions based on various metrics, such as readability, difficulty, word count, and similarity. The goal is to categorize and rank questions, aiding in the refinement of question banks for educational or assessment purposes.
Features

    Readability Analysis:
        Calculates the Flesch Reading Ease score to measure text complexity.
        Categorizes questions as Easy, Medium, or Hard based on readability.

    Difficulty Analysis:
        Computes difficulty as the ratio of successful students to total attendees.
        Categorizes questions as Easy, Medium, or Hard.

    Word Count Analysis:
        Counts the words in each question.
        Categorizes questions as Small, Medium, or Large based on word count.

    Similarity Analysis:
        Uses TF-IDF vectorization and cosine similarity to identify relationships between questions.
        Outputs the most similar question and its similarity score for each question.

    Ranking:
        Ranks questions by readability, difficulty, and word count.

Libraries Used

    pandas: Data manipulation and analysis.
    nltk: Tokenization and word count analysis.
    textstat: Readability score calculation.
    scikit-learn: TF-IDF vectorization and cosine similarity.

Input Data

The input data consists of a dictionary containing the following fields:

    Question: The text of the question.
    No of Attendees for that Question: The number of attendees for each question.
    No of Students Succeeded: The number of students who successfully answered the question.

Example:

data = {
    'Question': ['What is the capital of France?', 'Explain the theory of relativity in simple terms.', ...],
    'No of Attendees for that Question': [100, 90, ...],
    'No of Students Succeeded': [10, 30, ...]
}

Output

The script generates a detailed DataFrame with the following columns:

    Question: The original question text.
    ReadabilityScore: Flesch Reading Ease score.
    DifficultyScore: Ratio of successful students to total attendees.
    ReadabilityRank: Rank based on readability score.
    ReadabilityCategory: Categorization into Easy, Medium, or Hard.
    DifficultyRank: Rank based on difficulty score.
    DifficultyCategory: Categorization into Easy, Medium, or Hard.
    QuestionWordCount: Number of words in the question.
    QuestionRankByWords: Rank based on word count.
    QuestionSizebyWord: Categorization as Small, Medium, or Large Question.
    MostSimilarQuestion: The most similar question based on cosine similarity.
    SimilarityScore: Cosine similarity score of the most similar question.
